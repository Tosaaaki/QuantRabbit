"""Fail-closed scoring for AI-supervised DOJO bot training studies.

The trainer is deliberately split into three boundaries:

* candidate proposals and the study definition are content-addressed before a
  result is admitted;
* every candidate must supply the fixed ``OHLC/OLHC x BASE/STRESS`` cell grid;
* known TRAIN economics may be ranked, while missing continuous mark-to-market
  evidence remains an explicit promotion blocker.

Nothing in this module grants order authority, live permission, proof status,
or promotion.  It accepts only worn-history ``TRAIN`` studies.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import stat
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterator

from quant_rabbit.dojo_bot_catalog import (
    CATALOG_CONTRACT,
    DojoBotCatalogError,
    bot_config_risk_vector,
    bot_config_sha256,
    catalog_manifest,
    validate_bot_config,
)


PROPOSAL_CONTRACT = "QR_DOJO_BOT_CANDIDATE_PROPOSAL_V1"
STUDY_CONTRACT = "QR_DOJO_BOT_TRAINER_STUDY_V1"
SEALED_STUDY_CONTRACT = "QR_DOJO_BOT_TRAINER_SEALED_STUDY_V1"
CELL_CONTRACT = "QR_DOJO_BOT_TRAINER_CELL_V1"
EVALUATION_CONTRACT = "QR_DOJO_BOT_TRAINER_EVALUATION_V1"
LEDGER_METRICS_CONTRACT = "QR_DOJO_BOT_TRAINER_LEDGER_METRICS_V1"

MAX_CANDIDATES = 32
MAX_SOURCES = 256
MAX_JSON_BYTES = 8 * 1024 * 1024
REQUIRED_INTRABAR_PATHS = ("OHLC", "OLHC")
REQUIRED_COST_ARMS = ("BASE", "STRESS")
MIN_TRAIN_WINDOW_DAYS = 28.0
MIN_STRESS_SLIPPAGE_PIPS = 0.3
MIN_STRESS_FINANCING_PIPS_PER_DAY = 0.8
_ZERO_SHA256 = "0" * 64
_SHA256_LENGTH = 64
_MAX_LEDGER_BYTES = 4 * 1024 * 1024 * 1024
_MTM_COORDINATE_CONTRACT_V1 = "QR_REPLAY_MTM_COORDINATES_V1"
_MTM_COORDINATE_CONTRACT_V2 = "QR_REPLAY_ASYNC_MTM_COORDINATES_V2"
_MTM_COORDINATE_SCHEDULE_V2 = "SEALED_CORPUS_EPOCH_UNION"
_MTM_QUOTE_POLICY_V2 = "OBSERVED_ONLY_NO_SYNTHETIC_CARRY_QUOTES"
_MTM_MAX_CARRIED_QUOTE_AGE_SECONDS = 900
_EXPECTED_VIRTUAL_BROKER_LEVERAGE = 25.0
_QUOTE_BATCH_CONTRACT = "QR_VIRTUAL_QUOTE_BATCH_V1"
_ACCOUNT_MARK_CONTRACT = "QR_VIRTUAL_ACCOUNT_MARK_V1"
_FILL_EVENTS = {"FILL_MARKET", "FILL_LIMIT", "FILL_STOP"}
_EXIT_EVENTS = {"EXIT_TP", "EXIT_SL", "CLOSE", "MARGIN_CLOSEOUT"}
_MARGIN_REJECTION_EVENTS = {
    "ORDER_REJECTED_INSUFFICIENT_MARGIN",
    "LIMIT_REJECTED_INSUFFICIENT_MARGIN",
}
_OWNED_LEDGER_EVENTS = (
    _FILL_EVENTS
    | _EXIT_EVENTS
    | _MARGIN_REJECTION_EVENTS
    | {
        "ORDER_LIMIT",
        "ORDER_STOP",
        "ORDER_CANCEL",
        "ORDER_CANCEL_CONCURRENCY_CAP",
        "ORDER_REJECTED_CONCURRENCY_CAP",
        "SET_EXIT",
        "PERIOD_END_SETTLEMENT",
    }
)
_MAX_CORPUS_COMMITMENT_CACHE = 8
_CORPUS_COMMITMENT_CACHE: OrderedDict[str, str] = OrderedDict()


class DojoBotTrainerError(ValueError):
    """Raised when trainer inputs or evidence fail a closed contract."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DojoBotTrainerError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_jsonish(value: Any, *, field: str) -> Any:
    if isinstance(value, (str, bytes, bytearray)):
        raw = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        if len(raw) > MAX_JSON_BYTES:
            raise DojoBotTrainerError(f"{field} exceeds the JSON size limit")
        try:
            loaded = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
        except DojoBotTrainerError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise DojoBotTrainerError(f"{field} is not strict JSON") from exc
        _validate_json_tree(loaded, field=field)
        return loaded
    _validate_json_tree(value, field=field)
    return value


def _validate_json_tree(value: Any, *, field: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DojoBotTrainerError(f"{field} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise DojoBotTrainerError(f"{field} contains a non-string key")
            _validate_json_tree(item, field=f"{field}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _validate_json_tree(item, field=f"{field}[{index}]")
        return
    raise DojoBotTrainerError(f"{field} contains a non-JSON value")


def _canonical_sha256(value: Any) -> str:
    _validate_json_tree(value, field="canonical evidence")
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DojoBotTrainerError("evidence is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _require_exact_keys(
    value: Any, *, field: str, expected: set[str]
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise DojoBotTrainerError(f"{field} schema mismatch")
    return value


def _number(
    value: Any,
    *,
    field: str,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DojoBotTrainerError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise DojoBotTrainerError(f"{field} must be finite")
    if positive and number <= 0:
        raise DojoBotTrainerError(f"{field} must be positive")
    if non_negative and number < 0:
        raise DojoBotTrainerError(f"{field} must be non-negative")
    return number


def _integer(value: Any, *, field: str, non_negative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DojoBotTrainerError(f"{field} must be an integer")
    if non_negative and value < 0:
        raise DojoBotTrainerError(f"{field} must be non-negative")
    return value


def _identity(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise DojoBotTrainerError(f"{field} must contain 1..128 characters")
    if any(ord(character) < 33 or ord(character) > 126 for character in value):
        raise DojoBotTrainerError(f"{field} must be visible ASCII")
    return value


def _text(value: Any, *, field: str, maximum: int = 1024) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise DojoBotTrainerError(f"{field} must contain bounded non-empty text")
    if any(ord(character) < 32 and character not in "\n\t" for character in value):
        raise DojoBotTrainerError(f"{field} contains control characters")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DojoBotTrainerError(f"{field} must be a lowercase SHA-256")
    return value


def _pair(value: Any, *, field: str) -> str:
    pair = _identity(value, field=field)
    parts = pair.split("_")
    if len(parts) != 2 or any(
        len(part) != 3 or not part.isalpha() or not part.isupper() for part in parts
    ):
        raise DojoBotTrainerError(f"{field} is not an instrument pair")
    if parts[0] == parts[1]:
        raise DojoBotTrainerError(f"{field} is degenerate")
    return pair


def _normalize_source_digests(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value or len(value) > MAX_SOURCES:
        raise DojoBotTrainerError("source_digests must be a bounded non-empty object")
    normalized: dict[str, str] = {}
    for raw_path, raw_digest in value.items():
        if not isinstance(raw_path, str) or not raw_path:
            raise DojoBotTrainerError("source digest path is invalid")
        path = PurePosixPath(raw_path)
        if path.is_absolute() or ".." in path.parts or str(path) != raw_path:
            raise DojoBotTrainerError("source digest path must be safe and canonical")
        if raw_path in normalized:
            raise DojoBotTrainerError("duplicate source digest path")
        normalized[raw_path] = _sha256(raw_digest, field=f"source_digests[{raw_path}]")
    return dict(sorted(normalized.items()))


def _normalize_thresholds(value: Any, *, pair_count: int) -> dict[str, float]:
    row = _require_exact_keys(
        value,
        field="study.thresholds",
        expected={
            "normal_mtm_drawdown_max",
            "stress_mtm_drawdown_max",
            "peak_margin_usage_max",
            "margin_reject_rate_max",
            "cost_retention_min",
            "pair_positive_share_max",
            "pair_hhi_max",
        },
    )
    normalized = {
        key: _number(raw, field=f"study.thresholds.{key}", non_negative=True)
        for key, raw in row.items()
    }
    for key, number in normalized.items():
        if number > 1:
            raise DojoBotTrainerError(f"study.thresholds.{key} must be at most 1")
    if normalized["normal_mtm_drawdown_max"] <= 0:
        raise DojoBotTrainerError("normal MTM drawdown threshold must be positive")
    if normalized["stress_mtm_drawdown_max"] < normalized["normal_mtm_drawdown_max"]:
        raise DojoBotTrainerError("stress MTM drawdown threshold is too tight")
    if not 0 < normalized["peak_margin_usage_max"] <= 0.95:
        raise DojoBotTrainerError("peak margin threshold exceeds the absolute bound")
    if normalized["cost_retention_min"] <= 0:
        raise DojoBotTrainerError("cost retention threshold must be positive")
    equal_share = 1.0 / pair_count
    if normalized["pair_positive_share_max"] < equal_share:
        raise DojoBotTrainerError("pair share threshold is below equal allocation")
    if normalized["pair_hhi_max"] < equal_share:
        raise DojoBotTrainerError("pair HHI threshold is below its mathematical floor")
    hard_maximums = {
        "normal_mtm_drawdown_max": 0.10,
        "stress_mtm_drawdown_max": 0.15,
        "peak_margin_usage_max": 0.45,
        "margin_reject_rate_max": 0.10,
        "pair_positive_share_max": max(equal_share, 0.50),
        "pair_hhi_max": max(equal_share, 0.40),
    }
    for key, hard_maximum in hard_maximums.items():
        if normalized[key] > hard_maximum:
            raise DojoBotTrainerError(
                f"study.thresholds.{key} exceeds the repo-owned hard policy"
            )
    if normalized["cost_retention_min"] < 0.50:
        raise DojoBotTrainerError(
            "study.thresholds.cost_retention_min is below the repo-owned hard policy"
        )
    return normalized


def _normalize_window(value: Any) -> dict[str, Any]:
    row = _require_exact_keys(
        value,
        field="study.window",
        expected={
            "start_utc",
            "end_utc",
            "corpus_id",
            "corpus_sha256",
            "evidence_tier",
        },
    )
    if row.get("evidence_tier") != "WORN_TRAIN":
        raise DojoBotTrainerError("study window evidence tier must be WORN_TRAIN")
    if any("#" in str(row.get(field, "")) for field in ("start_utc", "end_utc")):
        raise DojoBotTrainerError("study window timestamps must be strict ISO-8601")
    start = _parse_utc(row.get("start_utc"), field="study.window.start_utc")
    end = _parse_utc(row.get("end_utc"), field="study.window.end_utc")
    if end <= start:
        raise DojoBotTrainerError("study window must satisfy start_utc < end_utc")
    if (end - start).total_seconds() < MIN_TRAIN_WINDOW_DAYS * 86_400:
        raise DojoBotTrainerError(
            f"study TRAIN window must span at least {MIN_TRAIN_WINDOW_DAYS:g} days"
        )
    return {
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "corpus_id": _identity(row.get("corpus_id"), field="study.window.corpus_id"),
        "corpus_sha256": _sha256(
            row.get("corpus_sha256"), field="study.window.corpus_sha256"
        ),
        "evidence_tier": "WORN_TRAIN",
    }


def _normalize_cost_arm(value: Any, *, arm: str) -> dict[str, float]:
    fields = (
        "slippage_pips_per_fill",
        "financing_pips_per_day",
        "recorded_spread_multiplier",
    )
    row = _require_exact_keys(
        value,
        field=f"study.cost_arms.{arm}",
        expected=set(fields),
    )
    return {
        key: _number(
            row[key],
            field=f"study.cost_arms.{arm}.{key}",
            non_negative=True,
        )
        for key in fields
    }


def _normalize_cost_arms(value: Any) -> dict[str, dict[str, float]]:
    row = _require_exact_keys(
        value,
        field="study.cost_arms",
        expected=set(REQUIRED_COST_ARMS),
    )
    base = _normalize_cost_arm(row["BASE"], arm="BASE")
    stress = _normalize_cost_arm(row["STRESS"], arm="STRESS")
    if base["recorded_spread_multiplier"] != 1.0:
        raise DojoBotTrainerError("BASE recorded spread multiplier must equal 1")
    if stress["recorded_spread_multiplier"] != 1.0:
        raise DojoBotTrainerError(
            "STRESS recorded spread multiplier must equal 1 until widening is implemented"
        )
    if any(stress[key] < base[key] for key in base):
        raise DojoBotTrainerError("STRESS costs must not be lower than BASE costs")
    if all(stress[key] == base[key] for key in base):
        raise DojoBotTrainerError("STRESS must be strictly harsher than BASE")
    if stress["slippage_pips_per_fill"] < MIN_STRESS_SLIPPAGE_PIPS:
        raise DojoBotTrainerError("STRESS slippage is below the repo-owned hard floor")
    if stress["financing_pips_per_day"] < MIN_STRESS_FINANCING_PIPS_PER_DAY:
        raise DojoBotTrainerError("STRESS financing is below the repo-owned hard floor")
    return {"BASE": base, "STRESS": stress}


def _normalize_proposer_evidence(value: Any) -> dict[str, Any]:
    row = _require_exact_keys(
        value,
        field="study.proposer_evidence",
        expected={
            "prompt_sha256",
            "input_sha256",
            "raw_response_sha256",
            "model_claim",
            "provider_attestation",
        },
    )
    attestation = row.get("provider_attestation")
    if attestation not in {"UNVERIFIED", "VERIFIED"}:
        raise DojoBotTrainerError("provider attestation is unsupported")
    return {
        "prompt_sha256": _sha256(
            row.get("prompt_sha256"), field="study.proposer_evidence.prompt_sha256"
        ),
        "input_sha256": _sha256(
            row.get("input_sha256"), field="study.proposer_evidence.input_sha256"
        ),
        "raw_response_sha256": _sha256(
            row.get("raw_response_sha256"),
            field="study.proposer_evidence.raw_response_sha256",
        ),
        "model_claim": _identity(
            row.get("model_claim"), field="study.proposer_evidence.model_claim"
        ),
        "provider_attestation": str(attestation),
    }


def _normalize_search_budget(value: Any) -> dict[str, int]:
    fields = ("attempt_ordinal", "total_attempts_in_lineage", "max_candidates")
    row = _require_exact_keys(
        value,
        field="study.search_budget",
        expected=set(fields),
    )
    normalized = {
        key: _integer(row[key], field=f"study.search_budget.{key}", non_negative=True)
        for key in fields
    }
    if any(number <= 0 for number in normalized.values()):
        raise DojoBotTrainerError("study search budget values must be positive")
    if normalized["attempt_ordinal"] > normalized["total_attempts_in_lineage"]:
        raise DojoBotTrainerError("attempt ordinal exceeds total attempts in lineage")
    if normalized["max_candidates"] > MAX_CANDIDATES:
        raise DojoBotTrainerError("search budget max_candidates exceeds hard cap")
    return normalized


def seal_candidate_proposal(proposal: Mapping[str, Any]) -> dict[str, Any]:
    """Seal one bounded candidate proposal before any result is inspected."""

    value = _load_jsonish(proposal, field="candidate proposal")
    if not isinstance(value, Mapping):
        raise DojoBotTrainerError("candidate proposal must be an object")
    expected_without_seal = {
        "contract",
        "schema_version",
        "candidate_id",
        "family",
        "hypothesis",
        "config",
        "risk_increase",
    }
    derived_fields = {
        "config_sha256",
        "catalog_contract",
        "catalog_sha256",
    }
    if set(value) == expected_without_seal | derived_fields | {"proposal_sha256"}:
        normalized = _normalize_proposal(value)
        return normalized
    row = _require_exact_keys(
        value,
        field="candidate proposal",
        expected=expected_without_seal,
    )
    body = _normalize_proposal_body(row)
    return {**body, "proposal_sha256": _canonical_sha256(body)}


def _normalize_proposal_body(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("contract") != PROPOSAL_CONTRACT or value.get("schema_version") != 1:
        raise DojoBotTrainerError("candidate proposal contract/version is unsupported")
    candidate_id = _identity(value.get("candidate_id"), field="candidate_id")
    family = _identity(value.get("family"), field="candidate.family")
    hypothesis = _text(value.get("hypothesis"), field="candidate.hypothesis")
    if value.get("risk_increase") is not False:
        raise DojoBotTrainerError("TRAIN candidate may not increase risk")
    try:
        normalized_config = validate_bot_config(value.get("config"))
        config_sha256 = bot_config_sha256(normalized_config)
        manifest = catalog_manifest()
    except DojoBotCatalogError as exc:
        raise DojoBotTrainerError("candidate bot config is invalid") from exc
    if family != normalized_config["signal"]:
        raise DojoBotTrainerError("candidate family does not match config signal")
    if manifest.get("contract") != CATALOG_CONTRACT:
        raise DojoBotTrainerError("bot catalog contract is unsupported")
    return {
        "contract": PROPOSAL_CONTRACT,
        "schema_version": 1,
        "candidate_id": candidate_id,
        "family": family,
        "hypothesis": hypothesis,
        "config": normalized_config,
        "config_sha256": config_sha256,
        "catalog_contract": CATALOG_CONTRACT,
        "catalog_sha256": _canonical_sha256(manifest),
        "risk_increase": False,
    }


def _normalize_proposal(value: Any) -> dict[str, Any]:
    row = _require_exact_keys(
        value,
        field="candidate proposal",
        expected={
            "contract",
            "schema_version",
            "candidate_id",
            "family",
            "hypothesis",
            "config",
            "config_sha256",
            "catalog_contract",
            "catalog_sha256",
            "risk_increase",
            "proposal_sha256",
        },
    )
    body = _normalize_proposal_body(
        {
            key: row[key]
            for key in (
                "contract",
                "schema_version",
                "candidate_id",
                "family",
                "hypothesis",
                "config",
                "risk_increase",
            )
        }
    )
    if (
        row["config_sha256"] != body["config_sha256"]
        or row["catalog_contract"] != body["catalog_contract"]
        or row["catalog_sha256"] != body["catalog_sha256"]
    ):
        raise DojoBotTrainerError("candidate catalog binding drift")
    claimed = _sha256(row["proposal_sha256"], field="candidate.proposal_sha256")
    if claimed != _canonical_sha256(body):
        raise DojoBotTrainerError("candidate proposal seal mismatch")
    return {**body, "proposal_sha256": claimed}


def _normalize_study(value: Any) -> dict[str, Any]:
    row = _require_exact_keys(
        value,
        field="study",
        expected={
            "contract",
            "schema_version",
            "study_id",
            "window_role",
            "initial_balance_jpy",
            "trade_pairs",
            "feed_pairs",
            "candidates",
            "thresholds",
            "window",
            "cost_arms",
            "proposer_evidence",
            "search_budget",
        },
    )
    if row.get("contract") != STUDY_CONTRACT or row.get("schema_version") != 1:
        raise DojoBotTrainerError("study contract/version is unsupported")
    if row.get("window_role") != "TRAIN":
        raise DojoBotTrainerError("bot trainer accepts TRAIN studies only")
    raw_pairs = row.get("trade_pairs")
    if not isinstance(raw_pairs, list) or len(raw_pairs) < 2 or len(raw_pairs) > 32:
        raise DojoBotTrainerError("study trade_pairs must contain 2..32 pairs")
    pairs = sorted(
        _pair(pair, field=f"study.trade_pairs[{index}]")
        for index, pair in enumerate(raw_pairs)
    )
    if len(set(pairs)) != len(pairs) or raw_pairs != pairs:
        raise DojoBotTrainerError("study trade_pairs must be unique and sorted")
    raw_feed_pairs = row.get("feed_pairs")
    if (
        not isinstance(raw_feed_pairs, list)
        or not raw_feed_pairs
        or len(raw_feed_pairs) > 32
    ):
        raise DojoBotTrainerError(
            "study feed_pairs must contain trade and conversion feeds"
        )
    feed_pairs = sorted(
        _pair(pair, field=f"study.feed_pairs[{index}]")
        for index, pair in enumerate(raw_feed_pairs)
    )
    if len(set(feed_pairs)) != len(feed_pairs) or raw_feed_pairs != feed_pairs:
        raise DojoBotTrainerError("study feed_pairs must be unique and sorted")
    if not set(pairs).issubset(feed_pairs):
        raise DojoBotTrainerError("study feed_pairs must include every trade pair")
    raw_candidates = row.get("candidates")
    if (
        not isinstance(raw_candidates, list)
        or not raw_candidates
        or len(raw_candidates) > MAX_CANDIDATES
    ):
        raise DojoBotTrainerError(
            f"study candidates must contain 1..{MAX_CANDIDATES} rows"
        )
    candidates = [_normalize_proposal(item) for item in raw_candidates]
    candidate_ids = [item["candidate_id"] for item in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise DojoBotTrainerError("study contains duplicate candidate ids")
    if candidate_ids != sorted(candidate_ids):
        raise DojoBotTrainerError("study candidates must be sorted by candidate id")
    if any(candidate["config"]["pairs"] != pairs for candidate in candidates):
        raise DojoBotTrainerError(
            "candidate config pairs must match study trade_pairs exactly"
        )
    search_budget = _normalize_search_budget(row.get("search_budget"))
    if len(candidates) > search_budget["max_candidates"]:
        raise DojoBotTrainerError("study candidates exceed the sealed search budget")
    return {
        "contract": STUDY_CONTRACT,
        "schema_version": 1,
        "study_id": _identity(row.get("study_id"), field="study.study_id"),
        "window_role": "TRAIN",
        "initial_balance_jpy": _number(
            row.get("initial_balance_jpy"),
            field="study.initial_balance_jpy",
            positive=True,
        ),
        "trade_pairs": pairs,
        "feed_pairs": feed_pairs,
        "candidates": candidates,
        "thresholds": _normalize_thresholds(
            row.get("thresholds"), pair_count=len(pairs)
        ),
        "window": _normalize_window(row.get("window")),
        "cost_arms": _normalize_cost_arms(row.get("cost_arms")),
        "proposer_evidence": _normalize_proposer_evidence(row.get("proposer_evidence")),
        "search_budget": search_budget,
    }


def seal_study(study: Mapping[str, Any], source_digests: Mapping[str, str]) -> dict:
    """Bind a strict TRAIN study to the source bytes used by its scorer."""

    normalized_study = _normalize_study(_load_jsonish(study, field="study"))
    normalized_sources = _normalize_source_digests(
        _load_jsonish(source_digests, field="source_digests")
    )
    body = {
        "contract": SEALED_STUDY_CONTRACT,
        "schema_version": 1,
        "study": normalized_study,
        "source_digests": normalized_sources,
        "classification": "WORN_HISTORICAL_TRAIN_ONLY",
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    return {**body, "study_sha256": _canonical_sha256(body)}


def _normalize_sealed_study(value: Any) -> dict[str, Any]:
    row = _require_exact_keys(
        value,
        field="sealed study",
        expected={
            "contract",
            "schema_version",
            "study",
            "source_digests",
            "classification",
            "proof_eligible",
            "promotion_eligible",
            "live_permission",
            "order_authority",
            "broker_mutation_allowed",
            "study_sha256",
        },
    )
    if row.get("contract") != SEALED_STUDY_CONTRACT or row.get("schema_version") != 1:
        raise DojoBotTrainerError("sealed study contract/version is unsupported")
    if (
        row.get("classification") != "WORN_HISTORICAL_TRAIN_ONLY"
        or row.get("proof_eligible") is not False
        or row.get("promotion_eligible") is not False
        or row.get("live_permission") is not False
        or row.get("order_authority") != "NONE"
        or row.get("broker_mutation_allowed") is not False
    ):
        raise DojoBotTrainerError("sealed study safety boundary mismatch")
    body = {
        "contract": SEALED_STUDY_CONTRACT,
        "schema_version": 1,
        "study": _normalize_study(row.get("study")),
        "source_digests": _normalize_source_digests(row.get("source_digests")),
        "classification": "WORN_HISTORICAL_TRAIN_ONLY",
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    claimed = _sha256(row.get("study_sha256"), field="sealed study SHA")
    if claimed != _canonical_sha256(body):
        raise DojoBotTrainerError("sealed study digest mismatch")
    return {**body, "study_sha256": claimed}


def verify_sealed_study(
    payload: Mapping[str, Any] | str | bytes,
    source_digests: Mapping[str, str],
) -> dict:
    """Verify the study seal and compare it to the current source digests."""

    normalized = _normalize_sealed_study(_load_jsonish(payload, field="sealed study"))
    current = _normalize_source_digests(
        _load_jsonish(source_digests, field="source_digests")
    )
    if current != normalized["source_digests"]:
        raise DojoBotTrainerError("source digest drift detected")
    return normalized


def _normalize_pair_values(
    value: Any, *, field: str, expected_pairs: Sequence[str]
) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(expected_pairs):
        raise DojoBotTrainerError(f"{field} must cover every expected pair exactly")
    return {
        pair: _number(value[pair], field=f"{field}.{pair}") for pair in expected_pairs
    }


def _artifact_relpath(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise DojoBotTrainerError(f"{field} must be a bounded relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or str(path) != value
        or path.name != "ledger.jsonl"
    ):
        raise DojoBotTrainerError(f"{field} must be a safe canonical ledger path")
    return value


def _normalize_ledger_receipt(
    value: Any, *, field: str, execution_status: str
) -> dict[str, Any]:
    row = _require_exact_keys(
        value,
        field=field,
        expected={
            "artifact_relpath",
            "artifact_size_bytes",
            "artifact_sha256",
            "ledger_terminal_sha256",
            "metrics_sha256",
            "corpus_sha256",
        },
    )
    artifact_relpath = row.get("artifact_relpath")
    if artifact_relpath is not None:
        artifact_relpath = _artifact_relpath(
            artifact_relpath, field=f"{field}.artifact_relpath"
        )
    artifact_size = _integer(
        row.get("artifact_size_bytes"),
        field=f"{field}.artifact_size_bytes",
        non_negative=True,
    )
    normalized = {
        "artifact_relpath": artifact_relpath,
        "artifact_size_bytes": artifact_size,
        "artifact_sha256": _sha256(
            row.get("artifact_sha256"), field=f"{field}.artifact_sha256"
        ),
        "ledger_terminal_sha256": _sha256(
            row.get("ledger_terminal_sha256"),
            field=f"{field}.ledger_terminal_sha256",
        ),
        "metrics_sha256": _sha256(
            row.get("metrics_sha256"), field=f"{field}.metrics_sha256"
        ),
        "corpus_sha256": _sha256(
            row.get("corpus_sha256"), field=f"{field}.corpus_sha256"
        ),
    }
    hashes = (
        normalized["artifact_sha256"],
        normalized["ledger_terminal_sha256"],
        normalized["metrics_sha256"],
        normalized["corpus_sha256"],
    )
    if execution_status == "SUCCESS":
        if artifact_relpath is None or artifact_size <= 0:
            raise DojoBotTrainerError(
                "successful cell ledger receipt must bind a non-empty artifact"
            )
        if any(digest == _ZERO_SHA256 for digest in hashes):
            raise DojoBotTrainerError(
                "successful cell may not use zero SHA ledger evidence"
            )
    elif (
        artifact_relpath is not None
        or artifact_size != 0
        or any(digest != _ZERO_SHA256 for digest in hashes)
    ):
        raise DojoBotTrainerError(
            "failed cell ledger receipts must use the empty evidence sentinel"
        )
    return normalized


def _normalize_ledger_evidence(
    value: Any, *, expected_pairs: Sequence[str], execution_status: str
) -> dict[str, Any]:
    row = _require_exact_keys(
        value,
        field="cell.ledger_evidence",
        expected={"main", "lopo_by_pair"},
    )
    lopo = row.get("lopo_by_pair")
    if not isinstance(lopo, Mapping) or set(lopo) != set(expected_pairs):
        raise DojoBotTrainerError(
            "cell.ledger_evidence.lopo_by_pair must cover every expected pair exactly"
        )
    normalized = {
        "main": _normalize_ledger_receipt(
            row.get("main"),
            field="cell.ledger_evidence.main",
            execution_status=execution_status,
        ),
        "lopo_by_pair": {
            pair: _normalize_ledger_receipt(
                lopo[pair],
                field=f"cell.ledger_evidence.lopo_by_pair.{pair}",
                execution_status=execution_status,
            )
            for pair in expected_pairs
        },
    }
    return normalized


def _normalize_cell_body(
    value: Any,
    *,
    sealed_study: Mapping[str, Any],
) -> dict[str, Any]:
    row = _require_exact_keys(
        value,
        field="training cell",
        expected={
            "contract",
            "schema_version",
            "study_sha256",
            "candidate_id",
            "proposal_sha256",
            "intrabar",
            "cost_arm",
            "metrics",
            "ledger_evidence",
            "execution_status",
            "failure_code",
        },
    )
    if row.get("contract") != CELL_CONTRACT or row.get("schema_version") != 1:
        raise DojoBotTrainerError("training cell contract/version is unsupported")
    study = sealed_study["study"]
    candidates = {item["candidate_id"]: item for item in study["candidates"]}
    candidate_id = _identity(row.get("candidate_id"), field="cell.candidate_id")
    if candidate_id not in candidates:
        raise DojoBotTrainerError("cell references an unknown candidate")
    proposal_sha = _sha256(row.get("proposal_sha256"), field="cell.proposal_sha256")
    if proposal_sha != candidates[candidate_id]["proposal_sha256"]:
        raise DojoBotTrainerError("cell candidate proposal digest drift")
    if row.get("study_sha256") != sealed_study["study_sha256"]:
        raise DojoBotTrainerError("cell study digest drift")
    intrabar = row.get("intrabar")
    cost_arm = row.get("cost_arm")
    if intrabar not in REQUIRED_INTRABAR_PATHS:
        raise DojoBotTrainerError("cell intrabar is unsupported")
    if cost_arm not in REQUIRED_COST_ARMS:
        raise DojoBotTrainerError("cell cost arm is unsupported")
    execution_status = row.get("execution_status")
    if execution_status not in {"SUCCESS", "FAILED"}:
        raise DojoBotTrainerError("cell execution_status is unsupported")
    failure_code = row.get("failure_code")
    if execution_status == "SUCCESS":
        if failure_code is not None:
            raise DojoBotTrainerError("successful cell failure_code must be null")
        normalized_failure_code = None
    else:
        normalized_failure_code = _identity(failure_code, field="cell.failure_code")
    metrics = _require_exact_keys(
        row.get("metrics"),
        field="cell.metrics",
        expected={
            "terminal_net_jpy",
            "terminal_flat",
            "margin_closeouts",
            "realized_max_drawdown_fraction",
            "mtm_complete",
            "mtm_max_drawdown_fraction",
            "peak_margin_usage_fraction",
            "fill_count",
            "margin_reject_count",
            "capital_lock_margin_jpy_hours",
            "pair_pnl_jpy",
            "leave_one_pair_out_net_jpy",
            "lopo_replay_complete",
        },
    )
    if any(
        not isinstance(metrics.get(field), bool)
        for field in ("terminal_flat", "mtm_complete", "lopo_replay_complete")
    ):
        raise DojoBotTrainerError("cell terminal/MTM/LOPO flags must be boolean")
    mtm_raw = metrics.get("mtm_max_drawdown_fraction")
    if metrics["mtm_complete"]:
        mtm_drawdown = _number(
            mtm_raw,
            field="cell.metrics.mtm_max_drawdown_fraction",
            non_negative=True,
        )
    elif mtm_raw is not None:
        raise DojoBotTrainerError("incomplete MTM evidence must use a null drawdown")
    else:
        mtm_drawdown = None
    expected_pairs = study["trade_pairs"]
    normalized_metrics = {
        "terminal_net_jpy": _number(
            metrics.get("terminal_net_jpy"), field="cell.metrics.terminal_net_jpy"
        ),
        "terminal_flat": metrics["terminal_flat"],
        "margin_closeouts": _integer(
            metrics.get("margin_closeouts"),
            field="cell.metrics.margin_closeouts",
            non_negative=True,
        ),
        "realized_max_drawdown_fraction": _number(
            metrics.get("realized_max_drawdown_fraction"),
            field="cell.metrics.realized_max_drawdown_fraction",
            non_negative=True,
        ),
        "mtm_complete": metrics["mtm_complete"],
        "mtm_max_drawdown_fraction": mtm_drawdown,
        "peak_margin_usage_fraction": _number(
            metrics.get("peak_margin_usage_fraction"),
            field="cell.metrics.peak_margin_usage_fraction",
            non_negative=True,
        ),
        "fill_count": _integer(
            metrics.get("fill_count"),
            field="cell.metrics.fill_count",
            non_negative=True,
        ),
        "margin_reject_count": _integer(
            metrics.get("margin_reject_count"),
            field="cell.metrics.margin_reject_count",
            non_negative=True,
        ),
        "capital_lock_margin_jpy_hours": _number(
            metrics.get("capital_lock_margin_jpy_hours"),
            field="cell.metrics.capital_lock_margin_jpy_hours",
            non_negative=True,
        ),
        "pair_pnl_jpy": _normalize_pair_values(
            metrics.get("pair_pnl_jpy"),
            field="cell.metrics.pair_pnl_jpy",
            expected_pairs=expected_pairs,
        ),
        "leave_one_pair_out_net_jpy": _normalize_pair_values(
            metrics.get("leave_one_pair_out_net_jpy"),
            field="cell.metrics.leave_one_pair_out_net_jpy",
            expected_pairs=expected_pairs,
        ),
        "lopo_replay_complete": metrics["lopo_replay_complete"],
    }
    tolerance = max(0.05, normalized_metrics["fill_count"] * 0.01)
    if not math.isclose(
        sum(normalized_metrics["pair_pnl_jpy"].values()),
        normalized_metrics["terminal_net_jpy"],
        rel_tol=0,
        abs_tol=tolerance,
    ):
        raise DojoBotTrainerError("cell pair PnL does not reconcile terminal net")
    return {
        "contract": CELL_CONTRACT,
        "schema_version": 1,
        "study_sha256": sealed_study["study_sha256"],
        "candidate_id": candidate_id,
        "proposal_sha256": proposal_sha,
        "intrabar": str(intrabar),
        "cost_arm": str(cost_arm),
        "execution_status": str(execution_status),
        "failure_code": normalized_failure_code,
        "metrics": normalized_metrics,
        "ledger_evidence": _normalize_ledger_evidence(
            row.get("ledger_evidence"),
            expected_pairs=expected_pairs,
            execution_status=str(execution_status),
        ),
    }


def _trusted_artifact_root(value: Path | str) -> Path:
    raw = Path(value)
    try:
        if raw.is_symlink():
            raise DojoBotTrainerError("artifact root may not be a symlink")
        root = raw.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DojoBotTrainerError("artifact root is missing or invalid") from exc
    if not root.is_dir():
        raise DojoBotTrainerError("artifact root must be a real directory")
    return root


@contextmanager
def _open_artifact(
    artifact_root: Path, receipt: Mapping[str, Any]
) -> Iterator[BinaryIO]:
    relative = PurePosixPath(str(receipt["artifact_relpath"]))
    current = artifact_root
    try:
        for part in relative.parts[:-1]:
            current = current / part
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise DojoBotTrainerError(
                    "ledger artifact path contains a symlink or non-directory"
                )
        target = current / relative.name
        info = target.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise DojoBotTrainerError("ledger artifact must be a regular non-symlink")
        if info.st_size > _MAX_LEDGER_BYTES:
            raise DojoBotTrainerError("ledger artifact exceeds the size limit")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(target, flags)
    except DojoBotTrainerError:
        raise
    except OSError as exc:
        raise DojoBotTrainerError("ledger artifact cannot be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != (
            info.st_dev,
            info.st_ino,
        ):
            raise DojoBotTrainerError("ledger artifact changed before verification")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            yield handle
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ):
            raise DojoBotTrainerError("ledger artifact changed during verification")
    finally:
        os.close(descriptor)


def _expected_owner_id(
    sealed_study: Mapping[str, Any],
    candidate_id: str,
    intrabar: str,
    cost_arm: str,
    held_out_pair: str | None,
) -> str:
    material = "|".join(
        [
            str(sealed_study["study_sha256"]),
            candidate_id,
            intrabar,
            cost_arm,
            held_out_pair or "MAIN",
        ]
    )
    return "dojo-trainer:" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def _expected_runtime_config_sha256(
    config: Mapping[str, Any], *, pairs: Sequence[str], owner_id: str
) -> str:
    runtime_input = json.loads(json.dumps(config, allow_nan=False))
    runtime_input["pairs"] = list(pairs)
    pair_cap = int(runtime_input["max_concurrent_per_pair"])
    runtime_input["global_max_concurrent"] = min(
        int(runtime_input["global_max_concurrent"]), pair_cap * len(pairs)
    )
    try:
        runtime = validate_bot_config(runtime_input)
    except DojoBotCatalogError as exc:
        raise DojoBotTrainerError("derived replay config violates bot catalog") from exc
    runtime["strategy_owner_id"] = owner_id
    encoded = json.dumps(
        runtime,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _score_verified_receipt(
    receipt: Mapping[str, Any],
    *,
    artifact_root: Path,
    sealed_study: Mapping[str, Any],
    candidate: Mapping[str, Any],
    intrabar: str,
    cost_arm: str,
    held_out_pair: str | None,
) -> dict[str, Any]:
    study = sealed_study["study"]
    trade_pairs = [pair for pair in study["trade_pairs"] if pair != held_out_pair]
    owner = _expected_owner_id(
        sealed_study,
        str(candidate["candidate_id"]),
        intrabar,
        cost_arm,
        held_out_pair,
    )
    config_sha = _expected_runtime_config_sha256(
        candidate["config"], pairs=trade_pairs, owner_id=owner
    )
    expected_pair_cap = int(candidate["config"]["max_concurrent_per_pair"])
    expected_global_cap = min(
        int(candidate["config"]["global_max_concurrent"]),
        expected_pair_cap * len(trade_pairs),
    )
    module_sha = sealed_study["source_digests"].get("bots/lab_bot.py")
    if module_sha is None:
        raise DojoBotTrainerError("sealed study does not bind bots/lab_bot.py")
    cost = study["cost_arms"][cost_arm]
    with _open_artifact(artifact_root, receipt) as handle:
        scored = score_ledger_metrics(
            handle,
            study["initial_balance_jpy"],
            trade_pairs,
            study["window"]["start_utc"],
            study["window"]["end_utc"],
            expected_intrabar=intrabar,
            expected_slippage_pips_per_fill=cost["slippage_pips_per_fill"],
            expected_financing_pips_per_day=cost["financing_pips_per_day"],
            expected_corpus_sha256=study["window"]["corpus_sha256"],
            expected_bot_config_sha256=config_sha,
            expected_strategy_owner_id=owner,
            expected_bot_module_sha256=module_sha,
            expected_bot_dependency_sha256=sealed_study["source_digests"],
            expected_feed_pairs=study["feed_pairs"],
            ledger_artifact_path=str(receipt["artifact_relpath"]),
            expected_max_concurrent_per_pair=expected_pair_cap,
            expected_global_max_concurrent=expected_global_cap,
        )
    for receipt_field, score_field in (
        ("artifact_size_bytes", "ledger_size_bytes"),
        ("artifact_sha256", "ledger_file_sha256"),
        ("ledger_terminal_sha256", "ledger_terminal_sha256"),
        ("metrics_sha256", "metrics_sha256"),
        ("corpus_sha256", "corpus_sha256"),
    ):
        if receipt[receipt_field] != scored[score_field]:
            raise DojoBotTrainerError(
                f"ledger receipt {receipt_field} does not match verified artifact"
            )
    return scored


def _cell_metrics_from_verified_ledgers(
    main: Mapping[str, Any],
    lopo: Mapping[str, Mapping[str, Any]],
    *,
    trade_pairs: Sequence[str],
) -> dict[str, Any]:
    if main.get("terminal_flat") is not True or any(
        row.get("terminal_flat") is not True for row in lopo.values()
    ):
        raise DojoBotTrainerError("successful cell ledgers must settle flat")
    return {
        "terminal_net_jpy": main["terminal_net_jpy"],
        "terminal_flat": main["terminal_flat"],
        "margin_closeouts": main["margin_closeouts"],
        "realized_max_drawdown_fraction": main["realized_max_drawdown_fraction"],
        "mtm_complete": main["mtm_complete"],
        "mtm_max_drawdown_fraction": main["mtm_max_drawdown_fraction"],
        "peak_margin_usage_fraction": main["peak_entry_margin_estimate_fraction"],
        "fill_count": main["fill_count"],
        "margin_reject_count": main["margin_reject_count"],
        "capital_lock_margin_jpy_hours": main["capital_lock_margin_jpy_hours"],
        "pair_pnl_jpy": {pair: main["pair_pnl_jpy"][pair] for pair in trade_pairs},
        "leave_one_pair_out_net_jpy": {
            pair: lopo[pair]["terminal_net_jpy"] for pair in trade_pairs
        },
        "lopo_replay_complete": True,
    }


def _verify_cell_artifacts(
    cell: Mapping[str, Any],
    *,
    sealed_study: Mapping[str, Any],
    artifact_root: Path,
) -> None:
    if cell["execution_status"] == "FAILED":
        return
    study = sealed_study["study"]
    candidate = {item["candidate_id"]: item for item in study["candidates"]}[
        cell["candidate_id"]
    ]
    receipts = cell["ledger_evidence"]
    paths = [receipts["main"]["artifact_relpath"]]
    paths.extend(
        receipts["lopo_by_pair"][pair]["artifact_relpath"]
        for pair in study["trade_pairs"]
    )
    if len(paths) != len(set(paths)):
        raise DojoBotTrainerError(
            "successful cell ledger artifact paths must be unique"
        )
    main = _score_verified_receipt(
        receipts["main"],
        artifact_root=artifact_root,
        sealed_study=sealed_study,
        candidate=candidate,
        intrabar=cell["intrabar"],
        cost_arm=cell["cost_arm"],
        held_out_pair=None,
    )
    lopo = {
        pair: _score_verified_receipt(
            receipts["lopo_by_pair"][pair],
            artifact_root=artifact_root,
            sealed_study=sealed_study,
            candidate=candidate,
            intrabar=cell["intrabar"],
            cost_arm=cell["cost_arm"],
            held_out_pair=pair,
        )
        for pair in study["trade_pairs"]
    }
    rebuilt = _cell_metrics_from_verified_ledgers(
        main, lopo, trade_pairs=study["trade_pairs"]
    )
    normalized_rebuilt = _normalize_cell_body(
        {
            **{key: value for key, value in cell.items() if key != "cell_sha256"},
            "metrics": rebuilt,
        },
        sealed_study=sealed_study,
    )["metrics"]
    if cell["metrics"] != normalized_rebuilt:
        raise DojoBotTrainerError(
            "cell metrics do not match metrics rebuilt from verified ledgers"
        )


def seal_cell_result(
    cell: Mapping[str, Any],
    sealed_study: Mapping[str, Any],
    *,
    artifact_root: Path | str,
) -> dict[str, Any]:
    """Seal a cell only after independently rebuilding its ledger evidence."""

    normalized_study = _normalize_sealed_study(sealed_study)
    trusted_root = _trusted_artifact_root(artifact_root)
    value = _load_jsonish(cell, field="training cell")
    if not isinstance(value, Mapping):
        raise DojoBotTrainerError("training cell must be an object")
    if "cell_sha256" in value:
        normalized = _normalize_cell(value, sealed_study=normalized_study)
        _verify_cell_artifacts(
            normalized, sealed_study=normalized_study, artifact_root=trusted_root
        )
        return normalized
    body = _normalize_cell_body(value, sealed_study=normalized_study)
    _verify_cell_artifacts(
        body, sealed_study=normalized_study, artifact_root=trusted_root
    )
    return {**body, "cell_sha256": _canonical_sha256(body)}


def _normalize_cell(value: Any, *, sealed_study: Mapping[str, Any]) -> dict[str, Any]:
    row = _require_exact_keys(
        value,
        field="sealed training cell",
        expected={
            "contract",
            "schema_version",
            "study_sha256",
            "candidate_id",
            "proposal_sha256",
            "intrabar",
            "cost_arm",
            "metrics",
            "ledger_evidence",
            "execution_status",
            "failure_code",
            "cell_sha256",
        },
    )
    body = _normalize_cell_body(
        {key: item for key, item in row.items() if key != "cell_sha256"},
        sealed_study=sealed_study,
    )
    claimed = _sha256(row.get("cell_sha256"), field="cell.cell_sha256")
    if claimed != _canonical_sha256(body):
        raise DojoBotTrainerError("training cell seal mismatch")
    return {**body, "cell_sha256": claimed}


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))


def _headroom(value: float, limit: float) -> float:
    if limit <= 0:
        return 1.0 if value <= 0 else 0.0
    return _clip(1.0 - value / limit)


def _pair_concentration(pair_pnl: Mapping[str, float]) -> tuple[float, float]:
    positive = [max(0.0, number) for number in pair_pnl.values()]
    total = sum(positive)
    if total <= 0:
        return 1.0, 1.0
    shares = [number / total for number in positive]
    return max(shares), sum(share * share for share in shares)


def _reject_rate(metrics: Mapping[str, Any]) -> float:
    fill_count = int(metrics["fill_count"])
    rejected = int(metrics["margin_reject_count"])
    attempts = fill_count + rejected
    return rejected / attempts if attempts else 1.0


def _effective_drawdown(metrics: Mapping[str, Any]) -> float:
    mtm = metrics["mtm_max_drawdown_fraction"]
    return float(mtm) if metrics["mtm_complete"] and mtm is not None else 1.0


def _score_candidate(
    candidate_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    sealed_study: Mapping[str, Any],
) -> dict[str, Any]:
    study = sealed_study["study"]
    threshold = study["thresholds"]
    start_balance = float(study["initial_balance_jpy"])
    pair_count = len(study["trade_pairs"])
    candidate = next(
        item for item in study["candidates"] if item["candidate_id"] == candidate_id
    )
    risk_vector = bot_config_risk_vector(
        candidate["config"],
        stress_slippage_pips_per_fill=study["cost_arms"]["STRESS"][
            "slippage_pips_per_fill"
        ],
    )

    metrics = [row["metrics"] for row in rows]
    failed_coordinates = sorted(
        f"{row['intrabar']}:{row['cost_arm']}:{row['failure_code']}"
        for row in rows
        if row["execution_status"] == "FAILED"
    )
    base = [row for row in rows if row["cost_arm"] == "BASE"]
    stress = [row for row in rows if row["cost_arm"] == "STRESS"]
    minimum_net = min(float(item["terminal_net_jpy"]) for item in metrics)
    realized_dd = max(float(item["realized_max_drawdown_fraction"]) for item in metrics)
    normal_dd = max(_effective_drawdown(row["metrics"]) for row in base)
    stress_dd = max(_effective_drawdown(row["metrics"]) for row in stress)
    peak_margin = max(float(item["peak_margin_usage_fraction"]) for item in metrics)
    reject_rate = max(_reject_rate(item) for item in metrics)
    mtm_complete = all(bool(item["mtm_complete"]) for item in metrics)
    lopo_replay_complete = all(bool(item["lopo_replay_complete"]) for item in metrics)

    concentration = [_pair_concentration(item["pair_pnl_jpy"]) for item in metrics]
    max_pair_share = max(item[0] for item in concentration)
    max_pair_hhi = max(item[1] for item in concentration)
    minimum_lopo = min(
        float(value)
        for item in metrics
        for value in item["leave_one_pair_out_net_jpy"].values()
    )

    by_coordinate = {(str(row["intrabar"]), str(row["cost_arm"])): row for row in rows}
    retention_by_path: dict[str, float | None] = {}
    for path in REQUIRED_INTRABAR_PATHS:
        base_net = float(by_coordinate[(path, "BASE")]["metrics"]["terminal_net_jpy"])
        stress_net = float(
            by_coordinate[(path, "STRESS")]["metrics"]["terminal_net_jpy"]
        )
        retention_by_path[path] = stress_net / base_net if base_net > 0 else None
    finite_retentions = [
        value for value in retention_by_path.values() if value is not None
    ]
    minimum_retention = (
        min(finite_retentions)
        if len(finite_retentions) == len(REQUIRED_INTRABAR_PATHS)
        else None
    )

    productivity_by_cell: dict[str, float | None] = {}
    for row in rows:
        lock = float(row["metrics"]["capital_lock_margin_jpy_hours"])
        net = float(row["metrics"]["terminal_net_jpy"])
        key = f"{row['intrabar']}:{row['cost_arm']}"
        productivity_by_cell[key] = net * 24.0 / lock if lock > 0 else None
    finite_productivity = [
        value for value in productivity_by_cell.values() if value is not None
    ]
    minimum_productivity = (
        min(finite_productivity) if len(finite_productivity) == len(rows) else None
    )

    blockers: list[str] = []
    blockers.extend(str(code) for code in risk_vector["blocker_codes"])
    if failed_coordinates:
        blockers.append("RUNNER_CELL_FAILURE")
    if not mtm_complete:
        blockers.append("CONTINUOUS_MTM_EVIDENCE_INCOMPLETE")
    if any(not item["terminal_flat"] for item in metrics):
        blockers.append("TERMINAL_EXPOSURE")
    if any(int(item["margin_closeouts"]) > 0 for item in metrics):
        blockers.append("MARGIN_CLOSEOUT_OCCURRED")
    if any(int(item["fill_count"]) <= 0 for item in metrics):
        blockers.append("ZERO_FILLS_IN_FIXED_CELL")
    if minimum_net <= 0:
        blockers.append("NON_POSITIVE_COORDINATE_WORST_NET")
    if normal_dd > threshold["normal_mtm_drawdown_max"]:
        blockers.append("NORMAL_DRAWDOWN_TOO_HIGH")
    if stress_dd > threshold["stress_mtm_drawdown_max"]:
        blockers.append("STRESS_DRAWDOWN_TOO_HIGH")
    if peak_margin > threshold["peak_margin_usage_max"]:
        blockers.append("PEAK_MARGIN_USAGE_TOO_HIGH")
    if reject_rate > threshold["margin_reject_rate_max"]:
        blockers.append("MARGIN_REJECT_RATE_TOO_HIGH")
    if minimum_retention is None or minimum_retention < threshold["cost_retention_min"]:
        blockers.append("COST_RETENTION_TOO_LOW")
    if max_pair_share > threshold["pair_positive_share_max"]:
        blockers.append("PAIR_POSITIVE_CONTRIBUTION_TOO_CONCENTRATED")
    if max_pair_hhi > threshold["pair_hhi_max"]:
        blockers.append("PAIR_CONTRIBUTION_HHI_TOO_HIGH")
    if minimum_lopo <= 0:
        blockers.append("LEAVE_ONE_PAIR_OUT_NOT_POSITIVE")
    if not lopo_replay_complete:
        blockers.append("COUNTERFACTUAL_LOPO_INCOMPLETE")
    if minimum_productivity is None:
        blockers.append("CAPITAL_LOCK_METRIC_INCOMPLETE")

    worst_return = minimum_net / start_balance
    return_quality = _clip(worst_return / max(stress_dd, 0.01))
    risk_quality = 0.5 * _headroom(
        normal_dd, threshold["normal_mtm_drawdown_max"]
    ) + 0.5 * _headroom(stress_dd, threshold["stress_mtm_drawdown_max"])
    margin_quality = _headroom(peak_margin, threshold["peak_margin_usage_max"])
    reject_quality = _headroom(reject_rate, threshold["margin_reject_rate_max"])
    if minimum_retention is None:
        cost_quality = 0.0
    else:
        retention_floor = threshold["cost_retention_min"]
        denominator = max(1.0 - retention_floor, 1e-12)
        cost_quality = _clip((minimum_retention - retention_floor) / denominator)
    equal_share = 1.0 / pair_count
    share_span = threshold["pair_positive_share_max"] - equal_share
    hhi_span = threshold["pair_hhi_max"] - equal_share
    share_quality = (
        _clip((threshold["pair_positive_share_max"] - max_pair_share) / share_span)
        if share_span > 0
        else float(max_pair_share <= threshold["pair_positive_share_max"])
    )
    hhi_quality = (
        _clip((threshold["pair_hhi_max"] - max_pair_hhi) / hhi_span)
        if hhi_span > 0
        else float(max_pair_hhi <= threshold["pair_hhi_max"])
    )
    concentration_quality = 0.5 * share_quality + 0.5 * hhi_quality
    productivity_quality = (
        _clip(minimum_productivity / (1.0 + minimum_productivity))
        if minimum_productivity is not None and minimum_productivity > 0
        else 0.0
    )
    score = 100.0 * (
        0.25 * return_quality
        + 0.20 * risk_quality
        + 0.15 * margin_quality
        + 0.10 * reject_quality
        + 0.10 * cost_quality
        + 0.10 * concentration_quality
        + 0.10 * productivity_quality
    )

    promotion_blockers = [
        "WORN_HISTORICAL_TRAIN_ONLY",
        "PROSPECTIVE_FORWARD_EVIDENCE_REQUIRED",
    ]
    if not mtm_complete:
        promotion_blockers.append("CONTINUOUS_MTM_EVIDENCE_INCOMPLETE")
    if not lopo_replay_complete:
        promotion_blockers.append("COUNTERFACTUAL_LOPO_INCOMPLETE")
    diagnostic_rank_eligible = not blockers
    return {
        "candidate_id": candidate_id,
        "status": (
            "TRAIN_DIAGNOSTIC_PASS" if diagnostic_rank_eligible else "TRAIN_REJECT"
        ),
        "diagnostic_rank_eligible": diagnostic_rank_eligible,
        "diagnostic_score": round(score, 6) if diagnostic_rank_eligible else None,
        "risk_policy_receipt": risk_vector,
        "proposer_risk_claim_ignored": True,
        "gate_blockers": blockers,
        "failed_coordinates": failed_coordinates,
        "coordinate_worst": {
            "terminal_net_jpy": round(minimum_net, 8),
            "realized_max_drawdown_fraction": round(realized_dd, 12),
            "normal_effective_drawdown_fraction": round(normal_dd, 12),
            "stress_effective_drawdown_fraction": round(stress_dd, 12),
            "peak_margin_usage_fraction": round(peak_margin, 12),
            "margin_reject_rate": round(reject_rate, 12),
            "cost_retention": (
                round(minimum_retention, 12) if minimum_retention is not None else None
            ),
            "pair_positive_share": round(max_pair_share, 12),
            "pair_hhi": round(max_pair_hhi, 12),
            "effective_positive_pairs": round(1.0 / max_pair_hhi, 12),
            "leave_one_pair_out_net_jpy": round(minimum_lopo, 8),
            "capital_productivity_per_margin_day": (
                round(minimum_productivity, 12)
                if minimum_productivity is not None
                else None
            ),
        },
        "cost_retention_by_intrabar": {
            key: round(value, 12) if value is not None else None
            for key, value in retention_by_path.items()
        },
        "capital_productivity_by_cell": {
            key: round(value, 12) if value is not None else None
            for key, value in sorted(productivity_by_cell.items())
        },
        "mtm_complete": mtm_complete,
        "mtm_incomplete_uses_realized_dd_for_train_diagnostic_only": False,
        "lopo_replay_complete": lopo_replay_complete,
        "promotion_gate_passed": False,
        "promotion_blockers": promotion_blockers,
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
    }


def evaluate_training(
    study: Mapping[str, Any] | str | bytes,
    cells: Sequence[Mapping[str, Any]] | str | bytes,
    *,
    artifact_root: Path | str,
) -> dict[str, Any]:
    """Re-open every cell ledger and evaluate a fixed TRAIN denominator."""

    sealed_study = _normalize_sealed_study(_load_jsonish(study, field="sealed study"))
    trusted_root = _trusted_artifact_root(artifact_root)
    raw_cells = _load_jsonish(cells, field="training cells")
    if not isinstance(raw_cells, list):
        raise DojoBotTrainerError("training cells must be a JSON array")
    expected_count = (
        len(sealed_study["study"]["candidates"])
        * len(REQUIRED_INTRABAR_PATHS)
        * len(REQUIRED_COST_ARMS)
    )
    if len(raw_cells) != expected_count:
        raise DojoBotTrainerError("training cells do not match the fixed denominator")
    normalized_cells = [
        _normalize_cell(item, sealed_study=sealed_study) for item in raw_cells
    ]
    for cell in normalized_cells:
        _verify_cell_artifacts(
            cell, sealed_study=sealed_study, artifact_root=trusted_root
        )
    coordinate_rows: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in normalized_cells:
        coordinate = (row["candidate_id"], row["intrabar"], row["cost_arm"])
        if coordinate in coordinate_rows:
            raise DojoBotTrainerError("duplicate fixed-denominator training cell")
        coordinate_rows[coordinate] = row
    expected_coordinates = {
        (candidate["candidate_id"], path, cost)
        for candidate in sealed_study["study"]["candidates"]
        for path in REQUIRED_INTRABAR_PATHS
        for cost in REQUIRED_COST_ARMS
    }
    if set(coordinate_rows) != expected_coordinates:
        raise DojoBotTrainerError("training cell coordinate grid is incomplete")

    evaluations = [
        _score_candidate(
            candidate["candidate_id"],
            [
                coordinate_rows[(candidate["candidate_id"], path, cost)]
                for path in REQUIRED_INTRABAR_PATHS
                for cost in REQUIRED_COST_ARMS
            ],
            sealed_study=sealed_study,
        )
        for candidate in sealed_study["study"]["candidates"]
    ]
    ranked = sorted(
        (row for row in evaluations if row["diagnostic_rank_eligible"]),
        key=lambda row: (
            -float(row["diagnostic_score"]),
            str(row["candidate_id"]),
        ),
    )
    body = {
        "contract": EVALUATION_CONTRACT,
        "schema_version": 1,
        "study_sha256": sealed_study["study_sha256"],
        "classification": "WORN_HISTORICAL_TRAIN_DIAGNOSTIC_ONLY",
        "fixed_denominator": {
            "candidate_count": len(evaluations),
            "intrabar_paths": list(REQUIRED_INTRABAR_PATHS),
            "cost_arms": list(REQUIRED_COST_ARMS),
            "expected_cell_count": expected_count,
            "observed_cell_count": len(normalized_cells),
            "coordinate_receipts_complete": True,
            "execution_success_complete": all(
                row["execution_status"] == "SUCCESS" for row in normalized_cells
            ),
        },
        "candidate_evaluations": sorted(
            evaluations, key=lambda row: str(row["candidate_id"])
        ),
        "diagnostic_ranking": [row["candidate_id"] for row in ranked],
        "rank_eligible_candidate_ids": [row["candidate_id"] for row in ranked],
        "unranked_candidate_ids": sorted(
            row["candidate_id"]
            for row in evaluations
            if not row["diagnostic_rank_eligible"]
        ),
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    return {**body, "evaluation_sha256": _canonical_sha256(body)}


def _parse_utc(value: Any, *, field: str, allow_naive: bool = False) -> datetime:
    if not isinstance(value, str) or not value:
        raise DojoBotTrainerError(f"{field} must be an ISO-8601 timestamp")
    text = value.split("#", 1)[0]
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise DojoBotTrainerError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        if not allow_naive:
            raise DojoBotTrainerError(f"{field} must include an explicit offset")
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _sim_quote_time(payload: Mapping[str, Any], *, field: str) -> datetime:
    quote = payload.get("quote")
    if not isinstance(quote, Mapping):
        raise DojoBotTrainerError(f"{field} quote evidence is missing")
    return _parse_utc(quote.get("ts"), field=f"{field}.quote.ts", allow_naive=True)


def _mtm_violation(detail: str) -> None:
    raise DojoBotTrainerError(f"MTM_CONTRACT_VIOLATION: {detail}")


def _mtm_price(pair: str, value: float) -> float:
    return round(value, 3 if pair.endswith("JPY") else 5)


def _mtm_pip(pair: str) -> float:
    return 0.01 if pair.endswith("JPY") else 0.0001


def _mtm_optional_number(value: Any, *, field: str) -> float | None:
    return None if value is None else _number(value, field=field, positive=True)


def _safe_corpus_shard_path(root: Path, relative: Any, *, field: str) -> Path:
    if not isinstance(relative, str) or not relative:
        _mtm_violation(f"{field} must be a non-empty relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        _mtm_violation(f"{field} escapes the sealed corpus root")
    if not root.is_absolute():
        _mtm_violation("manifest corpus root must be absolute")
    try:
        root_info = root.lstat()
    except OSError as exc:
        _mtm_violation(f"sealed corpus root is unavailable: {exc}")
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        _mtm_violation("sealed corpus root must be a real directory")
    current = root
    for part in pure.parts[:-1]:
        current /= part
        try:
            info = current.lstat()
        except OSError as exc:
            _mtm_violation(f"sealed corpus directory is unavailable: {exc}")
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            _mtm_violation("sealed corpus path contains a symlink or non-directory")
    path = current / pure.name
    try:
        info = path.lstat()
    except OSError as exc:
        _mtm_violation(f"sealed corpus shard is unavailable: {exc}")
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        _mtm_violation("sealed corpus shard must be a regular non-symlink")
    return path


def _verified_corpus_commitment_cache_key(
    *,
    corpus: Mapping[str, Any],
    replay: Mapping[str, Any],
    feed_pairs: Sequence[str],
    expected_batch_count: int,
) -> str:
    """Authenticate compressed shards and return a bounded-cache identity.

    Cache hits still hash every sealed compressed shard, so an in-place corpus
    mutation cannot inherit an earlier verification.  The cache only avoids
    repeating gzip inflation, strict JSON/OHLC parsing, and coordinate-chain
    derivation across candidate/cost cells sharing identical replay bytes.
    """

    root_value = corpus.get("root")
    raw_receipts = corpus.get("shards")
    if not isinstance(root_value, str) or not root_value:
        _mtm_violation("manifest corpus root is missing")
    if not isinstance(raw_receipts, list) or not raw_receipts:
        _mtm_violation("manifest corpus shards are missing")
    root = Path(root_value)
    authenticated: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for index, raw_receipt in enumerate(raw_receipts):
        field = f"manifest.corpus.shards[{index}]"
        receipt = _require_exact_keys(
            raw_receipt,
            field=field,
            expected={"path", "size_bytes", "sha256"},
        )
        relative = receipt["path"]
        if not isinstance(relative, str) or relative in seen_paths:
            _mtm_violation("manifest corpus shard paths must be unique strings")
        pair = PurePosixPath(relative).parent.name
        if pair not in feed_pairs:
            _mtm_violation("manifest corpus contains an unexpected pair shard")
        path = _safe_corpus_shard_path(root, relative, field=f"{field}.path")
        expected_size = _integer(
            receipt.get("size_bytes"), field=f"{field}.size_bytes", non_negative=True
        )
        expected_sha = _sha256(receipt.get("sha256"), field=f"{field}.sha256")
        try:
            flags = (
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(path, flags)
        except OSError as exc:
            _mtm_violation(f"sealed corpus shard cannot be opened safely: {exc}")
        with os.fdopen(descriptor, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size != expected_size:
                _mtm_violation(f"sealed corpus shard size changed: {path.name}")
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha:
            _mtm_violation(f"sealed corpus shard digest changed: {path.name}")
        seen_paths.add(relative)
        authenticated.append(
            {"path": relative, "size_bytes": expected_size, "sha256": expected_sha}
        )
    return _canonical_sha256(
        {
            "corpus_sha256": corpus.get("corpus_sha256"),
            "shards": authenticated,
            "time_from": replay.get("time_from"),
            "time_to": replay.get("time_to"),
            "granularity": replay.get("granularity"),
            "intrabar": replay.get("intrabar"),
            "feed_pairs": list(feed_pairs),
            "expected_batch_count": expected_batch_count,
            "mtm_coordinate_contract": replay.get("mtm_coordinate_contract"),
            "coordinate_schedule": replay.get("coordinate_schedule"),
            "quote_policy": replay.get("quote_policy"),
            "expected_union_epoch_count": replay.get("expected_union_epoch_count"),
            "expected_full_epoch_count": replay.get("expected_full_epoch_count"),
            "expected_partial_epoch_count": replay.get("expected_partial_epoch_count"),
            "expected_partial_phase_count": replay.get("expected_partial_phase_count"),
            "pair_row_counts": replay.get("pair_row_counts"),
            "availability_mask_sha256": replay.get("availability_mask_sha256"),
            "expected_quote_count": replay.get("expected_quote_count"),
            "max_carried_quote_age_seconds": replay.get(
                "max_carried_quote_age_seconds"
            ),
            "synthetic_quote_count": replay.get("synthetic_quote_count"),
        }
    )


def _iter_verified_pair_corpus_rows(
    *,
    root: Path,
    receipts: Sequence[Mapping[str, Any]],
    pair: str,
    granularity: str,
    window_start: datetime,
    window_end: datetime,
    maximum_rows: int,
) -> Iterator[tuple[int, dict[str, dict[str, float]]]]:
    previous_epoch: int | None = None
    yielded = 0
    pair_receipts = [
        receipt
        for receipt in receipts
        if PurePosixPath(str(receipt["path"])).parent.name == pair
    ]
    if not pair_receipts:
        _mtm_violation(f"sealed corpus has no shard for {pair}")

    def receipt_sort_key(receipt: Mapping[str, Any]) -> tuple[int, str]:
        path = str(receipt["path"])
        try:
            year = int(Path(path).name.split(f"_{granularity}_BA_", 1)[1][:4])
        except (IndexError, ValueError) as exc:
            raise DojoBotTrainerError(
                f"invalid sealed corpus shard name: {Path(path).name}"
            ) from exc
        return year, path

    pair_receipts.sort(key=receipt_sort_key)
    for receipt_index, receipt in enumerate(pair_receipts):
        field = f"manifest.corpus.shards[{receipt_index}]"
        path = _safe_corpus_shard_path(root, receipt["path"], field=f"{field}.path")
        size = _integer(
            receipt.get("size_bytes"), field=f"{field}.size_bytes", non_negative=True
        )
        expected_sha = _sha256(receipt.get("sha256"), field=f"{field}.sha256")
        expected_year = receipt_sort_key(receipt)[0]
        try:
            flags = (
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(path, flags)
        except OSError as exc:
            _mtm_violation(f"sealed corpus shard cannot be opened safely: {exc}")
        with os.fdopen(descriptor, "rb") as raw_handle:
            info = os.fstat(raw_handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size != size:
                _mtm_violation(f"sealed corpus shard size changed: {path.name}")
            digest = hashlib.sha256()
            for chunk in iter(lambda: raw_handle.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != expected_sha:
                _mtm_violation(f"sealed corpus shard digest changed: {path.name}")
            raw_handle.seek(0)
            try:
                gzip_handle = gzip.GzipFile(fileobj=raw_handle, mode="rb")
                with gzip_handle:
                    line_number = 0
                    while True:
                        raw_line = gzip_handle.readline(MAX_JSON_BYTES + 1)
                        if not raw_line:
                            break
                        line_number += 1
                        if len(raw_line) > MAX_JSON_BYTES:
                            _mtm_violation(
                                f"corpus row exceeds JSON limit: {path.name}:{line_number}"
                            )
                        row = _load_jsonish(
                            raw_line, field=f"corpus {path.name}:{line_number}"
                        )
                        if not isinstance(row, Mapping):
                            _mtm_violation(
                                f"corpus row must be an object: {path.name}:{line_number}"
                            )
                        stamp = _parse_utc(
                            row.get("time"),
                            field=f"corpus {path.name}:{line_number}.time",
                            allow_naive=True,
                        )
                        if stamp.year != expected_year:
                            _mtm_violation(
                                f"corpus row year/file mismatch: {path.name}:{line_number}"
                            )
                        parsed_sides: dict[str, dict[str, float]] = {}
                        for side in ("bid", "ask"):
                            values = row.get(side)
                            if not isinstance(values, Mapping) or not {
                                "o",
                                "h",
                                "l",
                                "c",
                            }.issubset(values):
                                _mtm_violation(
                                    f"corpus {side} OHLC is incomplete: "
                                    f"{path.name}:{line_number}"
                                )
                            parsed = {
                                key: _number(
                                    values[key],
                                    field=(
                                        f"corpus {path.name}:{line_number}."
                                        f"{side}.{key}"
                                    ),
                                    positive=True,
                                )
                                for key in ("o", "h", "l", "c")
                            }
                            if parsed["h"] < max(
                                parsed["o"], parsed["l"], parsed["c"]
                            ) or parsed["l"] > min(
                                parsed["o"], parsed["h"], parsed["c"]
                            ):
                                _mtm_violation(
                                    f"corpus {side} OHLC geometry is invalid: "
                                    f"{path.name}:{line_number}"
                                )
                            parsed_sides[side] = parsed
                        if any(
                            parsed_sides["ask"][key] < parsed_sides["bid"][key]
                            for key in ("o", "h", "l", "c")
                        ):
                            _mtm_violation(
                                f"corpus ask is below bid: {path.name}:{line_number}"
                            )
                        if not window_start <= stamp < window_end:
                            continue
                        epoch = int(stamp.timestamp())
                        if stamp.timestamp() != epoch or (
                            granularity == "M1" and epoch % 60 != 0
                        ):
                            _mtm_violation(
                                f"corpus row is not aligned to {granularity}: "
                                f"{path.name}:{line_number}"
                            )
                        if previous_epoch is not None and epoch <= previous_epoch:
                            _mtm_violation(
                                f"corpus rows are duplicate or noncausal for {pair}"
                            )
                        previous_epoch = epoch
                        yielded += 1
                        if yielded > maximum_rows:
                            _mtm_violation(
                                f"corpus has more rows than the MTM commitment for {pair}"
                            )
                        yield epoch, parsed_sides
            except (OSError, EOFError, UnicodeError) as exc:
                _mtm_violation(f"sealed corpus shard is corrupt: {path.name}: {exc}")
    if yielded == 0:
        _mtm_violation(f"sealed corpus has no in-window rows for {pair}")


def _iter_verified_corpus_batches(
    *,
    corpus: Mapping[str, Any],
    feed_pairs: Sequence[str],
    replay: Mapping[str, Any],
    expected_batch_count: int,
) -> Iterator[tuple[dict[str, Any], list[dict[str, Any]]]]:
    root_value = corpus.get("root")
    if not isinstance(root_value, str) or not root_value:
        _mtm_violation("manifest corpus root is missing")
    root = Path(root_value)
    raw_receipts = corpus.get("shards")
    if not isinstance(raw_receipts, list) or not raw_receipts:
        _mtm_violation("manifest corpus shards are missing")
    receipts: list[Mapping[str, Any]] = []
    seen_paths: set[str] = set()
    for index, raw_receipt in enumerate(raw_receipts):
        receipt = _require_exact_keys(
            raw_receipt,
            field=f"manifest.corpus.shards[{index}]",
            expected={"path", "size_bytes", "sha256"},
        )
        path = receipt["path"]
        if not isinstance(path, str) or path in seen_paths:
            _mtm_violation("manifest corpus shard paths must be unique strings")
        pair = PurePosixPath(path).parent.name
        if pair not in feed_pairs:
            _mtm_violation("manifest corpus contains an unexpected pair shard")
        seen_paths.add(path)
        receipts.append(receipt)
    window_start = _parse_utc(
        replay.get("time_from"), field="manifest.replay.time_from", allow_naive=True
    )
    window_end = _parse_utc(
        replay.get("time_to"), field="manifest.replay.time_to", allow_naive=True
    )
    phase_keys = (
        (("O", "o"), ("H", "h"), ("L", "l"), ("C", "c"))
        if replay.get("intrabar") == "OHLC"
        else (("O", "o"), ("L", "l"), ("H", "h"), ("C", "c"))
    )
    expected_epoch_count = expected_batch_count // len(phase_keys)
    sparse_union = replay.get("mtm_coordinate_contract") == _MTM_COORDINATE_CONTRACT_V2
    iterators = {
        pair: _iter_verified_pair_corpus_rows(
            root=root,
            receipts=receipts,
            pair=pair,
            granularity=str(replay.get("granularity")),
            window_start=window_start,
            window_end=window_end,
            maximum_rows=expected_epoch_count,
        )
        for pair in feed_pairs
    }
    sentinel = object()
    current: dict[str, Any] = {
        pair: next(iterator, sentinel) for pair, iterator in iterators.items()
    }
    epoch_count = 0
    while any(value is not sentinel for value in current.values()):
        if sparse_union:
            epoch = min(
                int(value[0]) for value in current.values() if value is not sentinel
            )
            batch_pairs = [
                pair
                for pair in feed_pairs
                if current[pair] is not sentinel and int(current[pair][0]) == epoch
            ]
        else:
            if any(value is sentinel for value in current.values()):
                _mtm_violation("sealed corpus feed pairs have partial epoch coverage")
            epochs = {int(value[0]) for value in current.values()}
            if len(epochs) != 1:
                _mtm_violation("sealed corpus feed pairs are not epoch-synchronized")
            epoch = epochs.pop()
            batch_pairs = list(feed_pairs)
        epoch_count += 1
        if epoch_count > expected_epoch_count:
            _mtm_violation("sealed corpus exceeds the committed coordinate count")
        for phase, price_key in phase_keys:
            timestamp = datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat() + (
                f"#{phase}"
            )
            coordinate = {
                "mode": "replay",
                "epoch": epoch,
                "phase": phase,
                "granularity": replay.get("granularity"),
                "intrabar": replay.get("intrabar"),
            }
            batch_quotes = [
                {
                    "pair": pair,
                    "bid": current[pair][1]["bid"][price_key],
                    "ask": current[pair][1]["ask"][price_key],
                    "ts": timestamp,
                }
                for pair in batch_pairs
            ]
            yield coordinate, batch_quotes
        if sparse_union:
            for pair in batch_pairs:
                current[pair] = next(iterators[pair], sentinel)
        else:
            current = {
                pair: next(iterator, sentinel) for pair, iterator in iterators.items()
            }
    if epoch_count != expected_epoch_count:
        _mtm_violation("sealed corpus has fewer rows than the MTM commitment")


def _verify_coordinate_mtm_contract(
    records: Iterator[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
    replay: Mapping[str, Any],
    feed_pairs: Sequence[str],
    trade_pairs: Sequence[str],
    start_balance: float,
    expected_owner: str,
    expected_intrabar: str,
    expected_slippage: float,
    expected_financing: float,
    leverage: float,
    expected_max_concurrent_per_pair: int | None,
    expected_global_max_concurrent: int | None,
) -> dict[str, Any] | None:
    """Verify the prospective coordinate-complete account-mark contract.

    Absence of the contract is the legacy fail-closed path.  Once declared,
    every coordinate, quote batch, broker mutation, state snapshot, account
    mark and terminal receipt is independently rebuilt; any defect rejects the
    ledger instead of silently falling back to diagnostic-only MTM.
    """

    if replay.get("mtm_coordinate_contract") is None:
        return None
    try:
        if (
            expected_max_concurrent_per_pair is None
            or expected_global_max_concurrent is None
        ):
            _mtm_violation(
                "owner concurrency caps are required for mandatory fill admission"
            )
        mtm_coordinate_contract = replay.get("mtm_coordinate_contract")
        if mtm_coordinate_contract not in {
            _MTM_COORDINATE_CONTRACT_V1,
            _MTM_COORDINATE_CONTRACT_V2,
        }:
            _mtm_violation("unknown replay MTM coordinate contract")
        sparse_union = mtm_coordinate_contract == _MTM_COORDINATE_CONTRACT_V2
        phase_order = replay.get("phase_order")
        expected_phase_order = (
            ["O", "H", "L", "C"]
            if expected_intrabar == "OHLC"
            else ["O", "L", "H", "C"]
        )
        if phase_order != expected_phase_order:
            _mtm_violation("manifest phase_order does not match intrabar mechanics")
        if replay.get("feed_pairs") != list(feed_pairs):
            _mtm_violation("manifest MTM feed_pairs are not exact and sorted")
        expected_batch_count = _integer(
            replay.get("expected_phase_mark_count"),
            field="manifest.replay.expected_phase_mark_count",
            non_negative=True,
        )
        if expected_batch_count <= 0:
            _mtm_violation("manifest expected phase mark count must be positive")
        if expected_batch_count % len(expected_phase_order) != 0:
            _mtm_violation("manifest phase mark count does not contain full bars")
        expected_union_epoch_count = expected_batch_count // len(expected_phase_order)
        expected_full_epoch_count = expected_union_epoch_count
        expected_partial_epoch_count = 0
        expected_partial_phase_count = 0
        expected_pair_row_counts = {
            pair: expected_union_epoch_count for pair in feed_pairs
        }
        expected_availability_mask_sha: str | None = None
        expected_quote_count = expected_batch_count * len(feed_pairs)
        max_carried_quote_age_seconds: int | None = None
        if sparse_union:
            if (
                replay.get("coordinate_schedule") != _MTM_COORDINATE_SCHEDULE_V2
                or replay.get("quote_policy") != _MTM_QUOTE_POLICY_V2
            ):
                _mtm_violation("manifest sparse coordinate/quote policy drifted")
            expected_union_epoch_count = _integer(
                replay.get("expected_union_epoch_count"),
                field="manifest.replay.expected_union_epoch_count",
                non_negative=True,
            )
            expected_full_epoch_count = _integer(
                replay.get("expected_full_epoch_count"),
                field="manifest.replay.expected_full_epoch_count",
                non_negative=True,
            )
            expected_partial_epoch_count = _integer(
                replay.get("expected_partial_epoch_count"),
                field="manifest.replay.expected_partial_epoch_count",
                non_negative=True,
            )
            expected_partial_phase_count = _integer(
                replay.get("expected_partial_phase_count"),
                field="manifest.replay.expected_partial_phase_count",
                non_negative=True,
            )
            if (
                expected_union_epoch_count <= 0
                or expected_union_epoch_count * len(expected_phase_order)
                != expected_batch_count
                or expected_full_epoch_count + expected_partial_epoch_count
                != expected_union_epoch_count
                or expected_partial_phase_count
                != expected_partial_epoch_count * len(expected_phase_order)
            ):
                _mtm_violation("manifest sparse coordinate counts are inconsistent")
            raw_pair_row_counts = replay.get("pair_row_counts")
            if not isinstance(raw_pair_row_counts, Mapping) or list(
                raw_pair_row_counts
            ) != list(feed_pairs):
                _mtm_violation("manifest sparse pair_row_counts are not exact/sorted")
            expected_pair_row_counts = {
                pair: _integer(
                    raw_pair_row_counts[pair],
                    field=f"manifest.replay.pair_row_counts.{pair}",
                    non_negative=True,
                )
                for pair in feed_pairs
            }
            if any(
                count <= 0 or count > expected_union_epoch_count
                for count in expected_pair_row_counts.values()
            ):
                _mtm_violation("manifest sparse pair row count is outside the union")
            expected_availability_mask_sha = _sha256(
                replay.get("availability_mask_sha256"),
                field="manifest.replay.availability_mask_sha256",
            )
            expected_quote_count = _integer(
                replay.get("expected_quote_count"),
                field="manifest.replay.expected_quote_count",
                non_negative=True,
            )
            if expected_quote_count != sum(expected_pair_row_counts.values()) * len(
                expected_phase_order
            ):
                _mtm_violation("manifest sparse expected_quote_count is inconsistent")
            max_carried_quote_age_seconds = _integer(
                replay.get("max_carried_quote_age_seconds"),
                field="manifest.replay.max_carried_quote_age_seconds",
                non_negative=True,
            )
            if (
                max_carried_quote_age_seconds != _MTM_MAX_CARRIED_QUOTE_AGE_SECONDS
                or _integer(
                    replay.get("synthetic_quote_count"),
                    field="manifest.replay.synthetic_quote_count",
                    non_negative=True,
                )
                != 0
            ):
                _mtm_violation("manifest sparse carry/synthetic quote policy drifted")
        elif (
            replay.get("full_pair_phase_coverage") is not True
            or _integer(
                replay.get("partial_phase_count"),
                field="manifest.replay.partial_phase_count",
                non_negative=True,
            )
            != 0
        ):
            _mtm_violation("manifest does not precommit full-pair phase coverage")
        expected_batch_terminal = _sha256(
            replay.get("expected_batch_chain_terminal_sha256"),
            field="manifest.replay.expected_batch_chain_terminal_sha256",
        )

        coordinate_keys = {"mode", "epoch", "phase", "granularity", "intrabar"}
        coordinate_window_start = _parse_utc(
            replay.get("time_from"),
            field="manifest.replay.time_from",
            allow_naive=True,
        )
        coordinate_window_end = _parse_utc(
            replay.get("time_to"),
            field="manifest.replay.time_to",
            allow_naive=True,
        )

        def coordinate(value: Any, *, field: str) -> dict[str, Any]:
            row = dict(
                _require_exact_keys(value, field=field, expected=coordinate_keys)
            )
            if row["mode"] != "replay":
                _mtm_violation(f"{field}.mode is not replay")
            epoch = _integer(row["epoch"], field=f"{field}.epoch", non_negative=True)
            if row["phase"] not in expected_phase_order:
                _mtm_violation(f"{field}.phase is not in the sealed phase order")
            if row["granularity"] != replay.get("granularity"):
                _mtm_violation(f"{field}.granularity drifted")
            if row["intrabar"] != expected_intrabar:
                _mtm_violation(f"{field}.intrabar drifted")
            coordinate_time = datetime.fromtimestamp(epoch, tz=timezone.utc)
            if not coordinate_window_start <= coordinate_time < coordinate_window_end:
                _mtm_violation(f"{field}.epoch is outside the sealed replay window")
            if row["granularity"] == "M1" and epoch % 60 != 0:
                _mtm_violation(f"{field}.epoch is not aligned to M1")
            return {**row, "epoch": epoch}

        expected_first_coordinate = coordinate(
            replay.get("first_coordinate"), field="manifest.replay.first_coordinate"
        )
        expected_last_coordinate = coordinate(
            replay.get("last_coordinate"), field="manifest.replay.last_coordinate"
        )
        if expected_first_coordinate["phase"] != "O":
            _mtm_violation("manifest first coordinate must begin at phase O")
        if expected_last_coordinate["phase"] != "C":
            _mtm_violation("manifest last coordinate must end at phase C")

        corpus = manifest.get("corpus")
        if not isinstance(corpus, Mapping):
            _mtm_violation("manifest corpus binding is missing")
        corpus_cache_key = _verified_corpus_commitment_cache_key(
            corpus=corpus,
            replay=replay,
            feed_pairs=feed_pairs,
            expected_batch_count=expected_batch_count,
        )
        cached_corpus_terminal = _CORPUS_COMMITMENT_CACHE.get(corpus_cache_key)
        if cached_corpus_terminal is not None:
            _CORPUS_COMMITMENT_CACHE.move_to_end(corpus_cache_key)
            if cached_corpus_terminal != expected_batch_terminal:
                _mtm_violation("cached corpus commitment conflicts with manifest")
            corpus_batches: (
                Iterator[tuple[dict[str, Any], list[dict[str, Any]]]] | None
            ) = None
        else:
            corpus_batches = _iter_verified_corpus_batches(
                corpus=corpus,
                feed_pairs=feed_pairs,
                replay=replay,
                expected_batch_count=expected_batch_count,
            )

        replay_identity_material = {
            "source": manifest.get("source"),
            "replay": manifest.get("replay"),
            "corpus": manifest.get("corpus"),
            "costs": manifest.get("costs"),
            "initial_balance_jpy": manifest.get("initial_balance_jpy"),
            "bot": manifest.get("bot"),
            "order_authority": manifest.get("order_authority"),
        }
        replay_identity_sha = _canonical_sha256(replay_identity_material)

        positions: dict[str, dict[str, Any]] = {}
        orders: dict[str, dict[str, Any]] = {}
        quotes: dict[str, dict[str, Any]] = {}
        quote_history: dict[str, list[dict[str, Any]]] = {}
        seen_trade_ids: set[str] = set()
        seen_order_ids: set[str] = set()
        reconstructed_balance = start_balance
        independently_resolved_exits = 0
        quote_sequence = 0
        batch_count = 0
        previous_batch_sha = _ZERO_SHA256
        previous_mark_sha = _ZERO_SHA256
        mark_count = 0
        first_observed_coordinate: dict[str, Any] | None = None
        last_observed_coordinate: dict[str, Any] | None = None
        current_batch: dict[str, Any] | None = None
        current_batch_manual_started = False
        current_market_consequence_trade_id: str | None = None
        current_market_consequence_window = False
        current_order_consequence_order_id: str | None = None
        current_order_consequence_window = False
        margin_closeout_sequence_active = False
        batch_eligible_order_ids: set[str] = set()
        batch_eligible_position_ids: set[str] = set()
        latest_epoch: int | None = None
        epoch_count = 0
        current_epoch_batch_pairs: list[str] | None = None
        observed_availability_mask: list[dict[str, Any]] = []
        observed_pair_row_counts = {pair: 0 for pair in feed_pairs}
        observed_full_epoch_count = 0
        observed_partial_epoch_count = 0
        mtm_equity_peak = start_balance
        mtm_drawdown = 0.0
        peak_margin_jpy = 0.0
        peak_margin_fraction = 0.0
        terminal_mark_payload: Mapping[str, Any] | None = None

        def current_quote(pair: str, *, field: str) -> dict[str, Any]:
            if pair not in quotes:
                _mtm_violation(f"{field} has no causally prior executable quote")
            return quotes[pair]

        def current_valuation_ts(*, field: str) -> str:
            batch = current_batch if current_batch is not None else latest_batch
            if batch is None:
                if not quotes:
                    _mtm_violation(f"{field} has no causal valuation timestamp")
                return str(
                    max(
                        quotes.values(),
                        key=lambda quote: _parse_utc(
                            quote["ts"],
                            field=f"{field}.quote.ts",
                            allow_naive=True,
                        ),
                    )["ts"]
                )
            coordinate_value = batch["coordinate"]
            return (
                datetime.fromtimestamp(
                    int(coordinate_value["epoch"]), tz=timezone.utc
                ).isoformat()
                + f"#{coordinate_value['phase']}"
            )

        def quote_as_of(
            pair: str, *, as_of_sequence: int, field: str
        ) -> dict[str, Any]:
            for quote in reversed(quote_history.get(pair, [])):
                if int(quote["sequence"]) <= as_of_sequence:
                    return quote
            _mtm_violation(f"{field} has no conversion quote at the causal watermark")

        def conversion_evidence(
            pair: str, *, reference_quote: Mapping[str, Any], field: str
        ) -> dict[str, Any]:
            as_of_sequence = _integer(
                reference_quote.get("watermark"),
                field=f"{field}.reference_quote.watermark",
                non_negative=True,
            )
            quote_currency = pair.split("_")[1]
            sources: list[tuple[str, dict[str, Any]]] = []
            if quote_currency == "JPY":
                rate = 1.0
            elif quote_currency == "USD":
                source = quote_as_of(
                    "USD_JPY", as_of_sequence=as_of_sequence, field=field
                )
                sources.append(("USD_JPY", source))
                rate = (float(source["bid"]) + float(source["ask"])) / 2.0
            else:
                direct_pair = f"{quote_currency}_JPY"
                try:
                    direct = quote_as_of(
                        direct_pair, as_of_sequence=as_of_sequence, field=field
                    )
                except DojoBotTrainerError as exc:
                    if not str(exc).startswith("MTM_CONTRACT_VIOLATION:"):
                        raise
                    direct = None
                if direct is not None:
                    sources.append((direct_pair, direct))
                    rate = (float(direct["bid"]) + float(direct["ask"])) / 2.0
                else:
                    usd_jpy = quote_as_of(
                        "USD_JPY", as_of_sequence=as_of_sequence, field=field
                    )
                    via_pair = f"USD_{quote_currency}"
                    via_usd = quote_as_of(
                        via_pair, as_of_sequence=as_of_sequence, field=field
                    )
                    via_mid = (float(via_usd["bid"]) + float(via_usd["ask"])) / 2.0
                    if via_mid <= 0:
                        _mtm_violation(
                            f"{field} conversion denominator is non-positive"
                        )
                    sources.extend((("USD_JPY", usd_jpy), (via_pair, via_usd)))
                    rate = (
                        (float(usd_jpy["bid"]) + float(usd_jpy["ask"])) / 2.0
                    ) / via_mid
            reference_time = _parse_utc(
                reference_quote.get("ts"),
                field=f"{field}.reference_quote.ts",
                allow_naive=True,
            )
            for source_pair, source in sources:
                source_time = _parse_utc(
                    source["ts"],
                    field=f"{field}.conversion_source.{source_pair}.ts",
                    allow_naive=True,
                )
                age_seconds = (reference_time - source_time).total_seconds()
                if age_seconds < 0:
                    _mtm_violation(f"{field} conversion quote is from the future")
                if (
                    max_carried_quote_age_seconds is not None
                    and age_seconds > max_carried_quote_age_seconds
                ):
                    _mtm_violation(f"{field} conversion quote exceeds sparse carry age")
            return {
                "quote_currency": quote_currency,
                "rate_jpy_per_quote_unit": rate,
                "as_of_quote_sequence": as_of_sequence,
                "source_quote_sequences": [
                    int(source["sequence"]) for _, source in sources
                ],
                "source_quotes": [
                    {
                        "pair": source_pair,
                        "bid": source["bid"],
                        "ask": source["ask"],
                        "ts": source["ts"],
                        "phase": (
                            str(source["ts"]).rsplit("#", 1)[1]
                            if "#" in str(source["ts"])
                            else None
                        ),
                    }
                    for source_pair, source in sources
                ],
            }

        def conversion_rate(
            pair: str, *, reference_quote: Mapping[str, Any], field: str
        ) -> float:
            return float(
                conversion_evidence(pair, reference_quote=reference_quote, field=field)[
                    "rate_jpy_per_quote_unit"
                ]
            )

        def financing_jpy(
            position: Mapping[str, Any], *, units: float, mark_ts: str, field: str
        ) -> float:
            opened = _parse_utc(
                position["opened_ts"], field=f"{field}.opened_ts", allow_naive=True
            )
            marked = _parse_utc(mark_ts, field=f"{field}.mark_ts", allow_naive=True)
            if marked < opened:
                _mtm_violation(f"{field} financing mark precedes the fill")
            held_days = (marked - opened).total_seconds() / 86400.0
            return (
                expected_financing
                * _mtm_pip(str(position["pair"]))
                * units
                * conversion_rate(
                    str(position["pair"]),
                    reference_quote=current_quote(str(position["pair"]), field=field),
                    field=field,
                )
                * held_days
            )

        def expected_account(*, field: str) -> dict[str, Any]:
            equity = reconstructed_balance
            accrued_financing = 0.0
            hedge_units: dict[str, dict[str, float]] = {}
            valuation_ts = current_valuation_ts(field=field) if positions else None
            for trade_id, position in positions.items():
                pair = str(position["pair"])
                quote = current_quote(pair, field=f"{field}.positions.{trade_id}")
                mark = (
                    float(quote["bid"])
                    if position["side"] == "LONG"
                    else float(quote["ask"])
                )
                direction_diff = (
                    mark - float(position["entry_price"])
                    if position["side"] == "LONG"
                    else float(position["entry_price"]) - mark
                )
                rate = conversion_rate(
                    pair,
                    reference_quote=quote,
                    field=f"{field}.positions.{trade_id}",
                )
                gross = direction_diff * float(position["units"]) * rate
                financing = financing_jpy(
                    position,
                    units=float(position["units"]),
                    mark_ts=str(valuation_ts),
                    field=f"{field}.positions.{trade_id}",
                )
                equity += gross - financing
                accrued_financing += financing
                sides = hedge_units.setdefault(pair, {"LONG": 0.0, "SHORT": 0.0})
                sides[str(position["side"])] += float(position["units"])
            margin = 0.0
            for pair, sides in hedge_units.items():
                quote = current_quote(pair, field=f"{field}.margin.{pair}")
                mid = (float(quote["bid"]) + float(quote["ask"])) / 2.0
                rate = conversion_rate(
                    pair, reference_quote=quote, field=f"{field}.margin.{pair}"
                )
                margin += max(sides["LONG"], sides["SHORT"]) * mid * rate / leverage
            usage = margin / equity if equity > 0 else 999.0
            return {
                "balance_jpy": round(reconstructed_balance, 2),
                "equity_jpy": round(equity, 2),
                "margin_used_jpy": round(margin, 2),
                "margin_usage": round(usage, 6),
                "accrued_financing_jpy": round(accrued_financing, 2),
                "open_positions": len(positions),
                "resting_orders": len(orders),
            }

        def touched_order(order: Mapping[str, Any], *, field: str) -> bool:
            quote = current_quote(str(order["pair"]), field=field)
            executable = float(
                quote["ask"] if order["side"] == "LONG" else quote["bid"]
            )
            protected = float(order["limit_price"])
            if order["kind"] == "LIMIT":
                return (
                    executable <= protected
                    if order["side"] == "LONG"
                    else executable >= protected
                )
            return (
                executable >= protected
                if order["side"] == "LONG"
                else executable <= protected
            )

        def touched_exit(position: Mapping[str, Any], *, field: str) -> str | None:
            quote = current_quote(str(position["pair"]), field=field)
            executable = float(
                quote["bid"] if position["side"] == "LONG" else quote["ask"]
            )
            stop = position["sl_price"]
            if stop is not None and (
                executable <= float(stop)
                if position["side"] == "LONG"
                else executable >= float(stop)
            ):
                return "EXIT_SL"
            target = position["tp_price"]
            if target is not None and (
                executable >= float(target)
                if position["side"] == "LONG"
                else executable <= float(target)
            ):
                return "EXIT_TP"
            return None

        def expected_concurrency_admission(
            order: Mapping[str, Any], *, field: str
        ) -> dict[str, Any] | None:
            pair = str(order["pair"])
            active_pair = sum(
                1 for position in positions.values() if position["pair"] == pair
            )
            active_global = len(positions)
            pair_cap = int(expected_max_concurrent_per_pair)
            global_cap = int(expected_global_max_concurrent)
            if active_pair >= pair_cap:
                scope = "PAIR"
                reason = "OWNER_PAIR_CONCURRENCY_CAP_REACHED"
            elif active_global >= global_cap:
                scope = "GLOBAL"
                reason = "OWNER_GLOBAL_CONCURRENCY_CAP_REACHED"
            else:
                return None
            return {
                "scope": scope,
                "reason": reason,
                "active_pair_positions": active_pair,
                "max_concurrent_per_pair": pair_cap,
                "active_global_positions": active_global,
                "global_max_concurrent": global_cap,
            }

        def margin_headroom_ok(order: Mapping[str, Any], *, field: str) -> bool:
            pair = str(order["pair"])
            side = str(order["side"])
            units = float(order["units"])
            if sparse_union and positions:
                valuation_time = _parse_utc(
                    current_valuation_ts(field=field),
                    field=f"{field}.valuation_ts",
                    allow_naive=True,
                )
                for trade_id, position in positions.items():
                    position_quote = current_quote(
                        str(position["pair"]), field=f"{field}.positions.{trade_id}"
                    )
                    quote_time = _parse_utc(
                        position_quote["ts"],
                        field=f"{field}.positions.{trade_id}.quote.ts",
                        allow_naive=True,
                    )
                    age_seconds = (valuation_time - quote_time).total_seconds()
                    if not 0 <= age_seconds <= _MTM_MAX_CARRIED_QUOTE_AGE_SECONDS:
                        return False
            account = expected_account(field=f"{field}.account")
            quote = current_quote(pair, field=f"{field}.quote")
            mid = (float(quote["bid"]) + float(quote["ask"])) / 2.0
            rate = conversion_rate(pair, reference_quote=quote, field=field)
            long_units = sum(
                float(position["units"])
                for position in positions.values()
                if position["pair"] == pair and position["side"] == "LONG"
            )
            short_units = sum(
                float(position["units"])
                for position in positions.values()
                if position["pair"] == pair and position["side"] == "SHORT"
            )
            old_pair_margin = max(long_units, short_units) * mid * rate / leverage
            if side == "LONG":
                long_units += units
            else:
                short_units += units
            new_pair_margin = max(long_units, short_units) * mid * rate / leverage
            new_total = (
                float(account["margin_used_jpy"]) - old_pair_margin + new_pair_margin
            )
            return new_total <= float(account["equity_jpy"])

        def resting_order_outcome(
            order_id: str, *, field: str
        ) -> tuple[str, Mapping[str, Any] | None]:
            order = orders[order_id]
            admission = expected_concurrency_admission(order, field=field)
            if admission is not None:
                return ("ORDER_CANCEL_CONCURRENCY_CAP", admission)
            if not margin_headroom_ok(order, field=field):
                return ("LIMIT_REJECTED_INSUFFICIENT_MARGIN", None)
            return ("FILL_LIMIT", None)

        def next_mandatory_broker_event(*, field: str) -> tuple[str, str] | None:
            """Derive the next broker-owned consequence at the staged batch.

            ``VirtualBroker.on_quote_batch`` processes canonical pairs, resting
            orders in insertion order, attached exits in position insertion
            order, and finally one portfolio margin closeout pass.  Replaying
            that ordering here prevents a ledger from silently keeping a
            touched loser or resting order alive until a favorable later bar.
            A nonmarketable order created by the O callback remains ineligible
            until the next quote batch.  An order already marketable at the
            staged O is made immediately eligible because the broker must
            resolve it atomically at that executable quote.  Market and resting
            fills are also made eligible for an immediate attached exit.
            """

            if current_batch is None:
                return None
            if margin_closeout_sequence_active and positions:
                return ("MARGIN_CLOSEOUT", next(iter(positions)))
            for pair in feed_pairs:
                for order_id, order in orders.items():
                    if (
                        order_id in batch_eligible_order_ids
                        and order["pair"] == pair
                        and touched_order(order, field=f"{field}.orders.{order_id}")
                    ):
                        expected_event, _ = resting_order_outcome(
                            order_id, field=f"{field}.orders.{order_id}"
                        )
                        return (expected_event, order_id)
                for trade_id, position in positions.items():
                    if (
                        trade_id not in batch_eligible_position_ids
                        or position["pair"] != pair
                    ):
                        continue
                    exit_event = touched_exit(
                        position, field=f"{field}.positions.{trade_id}"
                    )
                    if exit_event is not None:
                        return (exit_event, trade_id)
            account = expected_account(field=f"{field}.margin")
            if positions and float(account["margin_usage"]) >= 1.0:
                if sparse_union and not {
                    str(position["pair"]) for position in positions.values()
                }.issubset(set(current_batch["batch_pairs"])):
                    _mtm_violation(
                        f"{field} margin closeout lacks fresh position quotes"
                    )
                return ("MARGIN_CLOSEOUT", next(iter(positions)))
            return None

        def require_mandatory_event_matches(
            event: str, payload: Mapping[str, Any], *, field: str
        ) -> None:
            mandatory = next_mandatory_broker_event(field=field)
            if mandatory is None:
                _mtm_violation(f"{field} has no causally mandatory broker event")
            expected_event, identity = mandatory
            if expected_event in {
                "FILL_LIMIT",
                "LIMIT_REJECTED_INSUFFICIENT_MARGIN",
                "ORDER_CANCEL_CONCURRENCY_CAP",
            }:
                if event != expected_event or payload.get("order_id") != identity:
                    _mtm_violation(
                        f"{field} bypasses the next touched resting-order consequence"
                    )
                if expected_event == "ORDER_CANCEL_CONCURRENCY_CAP":
                    _, expected_admission = resting_order_outcome(
                        identity, field=f"{field}.admission"
                    )
                    if payload.get("admission") != expected_admission:
                        _mtm_violation(
                            f"{field}.admission does not match owner concurrency state"
                        )
                return
            if event != expected_event or payload.get("trade_id") != identity:
                _mtm_violation(
                    f"{field} bypasses mandatory {expected_event} for {identity}"
                )

        def require_no_mandatory_event(*, field: str) -> None:
            mandatory = next_mandatory_broker_event(field=field)
            if mandatory is not None:
                event, identity = mandatory
                _mtm_violation(
                    f"{field} omits mandatory broker consequence {event}:{identity}"
                )

        def validate_conversion(
            payload: Mapping[str, Any],
            pair: str,
            *,
            reference_quote: Mapping[str, Any],
            field: str,
        ) -> None:
            evidence = payload.get("conversion")
            if not isinstance(evidence, Mapping):
                _mtm_violation(f"{field}.conversion is missing")
            expected = conversion_evidence(
                pair, reference_quote=reference_quote, field=field
            )
            if dict(evidence) != expected:
                _mtm_violation(f"{field} conversion evidence is not quote-derived")

        def validate_event_quote(
            payload: Mapping[str, Any], pair: str, *, field: str
        ) -> dict[str, Any]:
            observed = payload.get("quote")
            if not isinstance(observed, Mapping) or set(observed) != {
                "bid",
                "ask",
                "ts",
            }:
                _mtm_violation(f"{field}.quote schema mismatch")
            expected = current_quote(pair, field=field)
            comparable = {key: expected[key] for key in ("bid", "ask", "ts")}
            if dict(observed) != comparable:
                _mtm_violation(f"{field}.quote is not the current batch quote")
            return expected

        def add_order(event: str, payload: Mapping[str, Any], *, field: str) -> None:
            order_id = _identity(payload.get("order_id"), field=f"{field}.order_id")
            if order_id in seen_order_ids or order_id in orders:
                _mtm_violation(f"{field} reuses an order id")
            pair = _pair(payload.get("pair"), field=f"{field}.pair")
            if pair not in feed_pairs:
                _mtm_violation(f"{field}.pair is outside the sealed feed")
            side = payload.get("side")
            if side not in {"LONG", "SHORT"}:
                _mtm_violation(f"{field}.side is invalid")
            units = _number(payload.get("units"), field=f"{field}.units", positive=True)
            price = _number(payload.get("price"), field=f"{field}.price", positive=True)
            tp_pips = _mtm_optional_number(
                payload.get("tp_pips"), field=f"{field}.tp_pips"
            )
            sl_pips = _mtm_optional_number(
                payload.get("sl_pips"), field=f"{field}.sl_pips"
            )
            orders[order_id] = {
                "order_id": order_id,
                "pair": pair,
                "side": side,
                "units": units,
                "limit_price": price,
                "tp_pips": tp_pips,
                "sl_pips": sl_pips,
                "kind": "STOP" if event == "ORDER_STOP" else "LIMIT",
            }
            seen_order_ids.add(order_id)

        def add_fill(event: str, payload: Mapping[str, Any], *, field: str) -> None:
            nonlocal reconstructed_balance
            if (
                _number(
                    payload.get("slippage_pips"),
                    field=f"{field}.slippage_pips",
                    non_negative=True,
                )
                != expected_slippage
            ):
                _mtm_violation(f"{field}.slippage_pips drifted")
            trade_id = _identity(payload.get("trade_id"), field=f"{field}.trade_id")
            if trade_id in seen_trade_ids or trade_id in positions:
                _mtm_violation(f"{field} reuses a trade id")
            pair = _pair(payload.get("pair"), field=f"{field}.pair")
            if pair not in trade_pairs:
                _mtm_violation(f"{field}.pair is not a sealed trade pair")
            side = payload.get("side")
            if side not in {"LONG", "SHORT"}:
                _mtm_violation(f"{field}.side is invalid")
            units = _number(payload.get("units"), field=f"{field}.units", positive=True)
            quote = validate_event_quote(payload, pair, field=field)
            validate_conversion(payload, pair, reference_quote=quote, field=field)
            pip = _mtm_pip(pair)
            if event == "FILL_MARKET":
                proposed_entry = {"pair": pair, "side": side, "units": units}
                if (
                    expected_concurrency_admission(
                        proposed_entry, field=f"{field}.admission"
                    )
                    is not None
                ):
                    _mtm_violation(f"{field} bypasses owner concurrency admission")
                if not margin_headroom_ok(
                    proposed_entry, field=f"{field}.margin_admission"
                ):
                    _mtm_violation(f"{field} bypasses market margin admission")
                raw_entry = (
                    float(quote["ask"]) + expected_slippage * pip
                    if side == "LONG"
                    else float(quote["bid"]) - expected_slippage * pip
                )
                expected_entry = _mtm_price(pair, raw_entry)
                entry = _number(
                    payload.get("entry"), field=f"{field}.entry", positive=True
                )
                if entry != expected_entry:
                    _mtm_violation(f"{field}.entry is not the executable market fill")
                tp = _mtm_optional_number(payload.get("tp"), field=f"{field}.tp")
                sl = _mtm_optional_number(payload.get("sl"), field=f"{field}.sl")
            else:
                order_id = _identity(payload.get("order_id"), field=f"{field}.order_id")
                pending = orders.get(order_id)
                if pending is None:
                    _mtm_violation(f"{field} has no causally prior resting order")
                if (
                    pending["pair"] != pair
                    or pending["side"] != side
                    or float(pending["units"]) != units
                ):
                    _mtm_violation(f"{field} does not match its resting order")
                order_kind = payload.get("order_kind")
                if order_kind != pending["kind"]:
                    _mtm_violation(f"{field}.order_kind does not match the order")
                protected = float(pending["limit_price"])
                executable = float(quote["ask"] if side == "LONG" else quote["bid"])
                if order_kind == "LIMIT":
                    touched = (
                        executable <= protected
                        if side == "LONG"
                        else executable >= protected
                    )
                    raw_entry = (
                        min(protected, executable)
                        if side == "LONG"
                        else max(protected, executable)
                    )
                else:
                    touched = (
                        executable >= protected
                        if side == "LONG"
                        else executable <= protected
                    )
                    raw_entry = (
                        max(protected, executable)
                        if side == "LONG"
                        else min(protected, executable)
                    )
                if not touched:
                    _mtm_violation(f"{field} fills an untouched resting order")
                if expected_slippage > 0:
                    stressed = _mtm_price(
                        pair,
                        raw_entry + expected_slippage * pip
                        if side == "LONG"
                        else raw_entry - expected_slippage * pip,
                    )
                    expected_entry = (
                        min(protected, stressed)
                        if order_kind == "LIMIT" and side == "LONG"
                        else max(protected, stressed)
                        if order_kind == "LIMIT"
                        else stressed
                    )
                else:
                    expected_entry = raw_entry
                entry = _number(
                    payload.get("entry"), field=f"{field}.entry", positive=True
                )
                price_alias = _number(
                    payload.get("price"), field=f"{field}.price", positive=True
                )
                if entry != expected_entry or price_alias != expected_entry:
                    _mtm_violation(f"{field}.entry is not the resting-order fill")
                expected_tp = (
                    _mtm_price(
                        pair,
                        entry + float(pending["tp_pips"]) * pip
                        if side == "LONG"
                        else entry - float(pending["tp_pips"]) * pip,
                    )
                    if pending["tp_pips"] is not None
                    else None
                )
                expected_sl = (
                    _mtm_price(
                        pair,
                        entry - float(pending["sl_pips"]) * pip
                        if side == "LONG"
                        else entry + float(pending["sl_pips"]) * pip,
                    )
                    if pending["sl_pips"] is not None
                    else None
                )
                tp = _mtm_optional_number(payload.get("tp"), field=f"{field}.tp")
                sl = _mtm_optional_number(payload.get("sl"), field=f"{field}.sl")
                if tp != expected_tp or sl != expected_sl:
                    _mtm_violation(f"{field} attached exits drifted from the order")
                del orders[order_id]
            positions[trade_id] = {
                "trade_id": trade_id,
                "pair": pair,
                "side": side,
                "units": units,
                "entry_price": entry,
                "opened_ts": str(quote["ts"]),
                "tp_price": tp,
                "sl_price": sl,
            }
            seen_trade_ids.add(trade_id)

        def resolve_exit(event: str, payload: Mapping[str, Any], *, field: str) -> None:
            nonlocal reconstructed_balance, independently_resolved_exits
            if (
                _number(
                    payload.get("slippage_pips"),
                    field=f"{field}.slippage_pips",
                    non_negative=True,
                )
                != expected_slippage
            ):
                _mtm_violation(f"{field}.slippage_pips drifted")
            trade_id = _identity(payload.get("trade_id"), field=f"{field}.trade_id")
            position = positions.get(trade_id)
            if position is None:
                _mtm_violation(f"{field} has no active trade")
            pair = str(position["pair"])
            quote = validate_event_quote(payload, pair, field=field)
            validate_conversion(payload, pair, reference_quote=quote, field=field)
            close_units = (
                _number(payload.get("units"), field=f"{field}.units", positive=True)
                if event == "CLOSE"
                else float(position["units"])
            )
            if close_units > float(position["units"]) + 1e-9:
                _mtm_violation(f"{field} over-closes its trade")
            pip = _mtm_pip(pair)
            executable = float(
                quote["bid"] if position["side"] == "LONG" else quote["ask"]
            )
            if event == "EXIT_SL":
                stop = position["sl_price"]
                if stop is None:
                    _mtm_violation(f"{field} has no attached stop")
                touched = (
                    executable <= float(stop)
                    if position["side"] == "LONG"
                    else executable >= float(stop)
                )
                raw_price = (
                    min(float(stop), executable)
                    if position["side"] == "LONG"
                    else max(float(stop), executable)
                )
                if not touched:
                    _mtm_violation(f"{field} resolves an untouched stop")
                expected_price = _mtm_price(
                    pair,
                    raw_price - expected_slippage * pip
                    if position["side"] == "LONG"
                    else raw_price + expected_slippage * pip,
                )
            elif event == "EXIT_TP":
                target = position["tp_price"]
                if target is None:
                    _mtm_violation(f"{field} has no attached target")
                stop = position["sl_price"]
                if stop is not None and (
                    executable <= float(stop)
                    if position["side"] == "LONG"
                    else executable >= float(stop)
                ):
                    _mtm_violation(f"{field} bypasses pessimistic SL-first ordering")
                touched = (
                    executable >= float(target)
                    if position["side"] == "LONG"
                    else executable <= float(target)
                )
                if not touched:
                    _mtm_violation(f"{field} resolves an untouched target")
                expected_price = _mtm_price(pair, float(target))
            else:
                expected_price = _mtm_price(
                    pair,
                    executable - expected_slippage * pip
                    if position["side"] == "LONG"
                    else executable + expected_slippage * pip,
                )
            observed_price = _number(
                payload.get("price"), field=f"{field}.price", positive=True
            )
            if observed_price != expected_price:
                _mtm_violation(f"{field}.price is not independently executable")
            rate = conversion_rate(pair, reference_quote=quote, field=field)
            gross = (
                (
                    (expected_price - float(position["entry_price"]))
                    if position["side"] == "LONG"
                    else (float(position["entry_price"]) - expected_price)
                )
                * close_units
                * rate
            )
            financing = financing_jpy(
                position,
                units=close_units,
                mark_ts=(
                    current_valuation_ts(field=field)
                    if event == "MARGIN_CLOSEOUT"
                    else str(quote["ts"])
                ),
                field=field,
            )
            expected_pl = gross - financing
            observed_pl = _number(payload.get("pl_jpy"), field=f"{field}.pl_jpy")
            if not math.isclose(
                observed_pl, round(expected_pl, 2), rel_tol=0, abs_tol=0.01
            ):
                _mtm_violation(f"{field}.pl_jpy is not independently reconstructed")
            if "gross_pl_jpy" in payload and not math.isclose(
                _number(payload["gross_pl_jpy"], field=f"{field}.gross_pl_jpy"),
                round(gross, 2),
                rel_tol=0,
                abs_tol=0.01,
            ):
                _mtm_violation(f"{field}.gross_pl_jpy is inconsistent")
            if "financing_jpy" in payload and not math.isclose(
                _number(
                    payload["financing_jpy"],
                    field=f"{field}.financing_jpy",
                    non_negative=True,
                ),
                round(financing, 2),
                rel_tol=0,
                abs_tol=0.01,
            ):
                _mtm_violation(f"{field}.financing_jpy is inconsistent")
            reconstructed_balance += expected_pl
            position["units"] = float(position["units"]) - close_units
            if float(position["units"]) <= 1e-9:
                del positions[trade_id]
            independently_resolved_exits += 1

        def apply_action(event: str, payload: Mapping[str, Any], *, field: str) -> None:
            if event in {"ORDER_LIMIT", "ORDER_STOP"}:
                add_order(event, payload, field=field)
            elif event in _FILL_EVENTS:
                if event == "FILL_STOP":
                    _mtm_violation(
                        f"{field} uses a non-producer fill event; STOP orders emit FILL_LIMIT"
                    )
                add_fill(event, payload, field=field)
            elif event in _EXIT_EVENTS:
                resolve_exit(event, payload, field=field)
            elif event in {
                "ORDER_CANCEL",
                "ORDER_CANCEL_CONCURRENCY_CAP",
                "LIMIT_REJECTED_INSUFFICIENT_MARGIN",
            }:
                order_id = _identity(payload.get("order_id"), field=f"{field}.order_id")
                if order_id not in orders:
                    _mtm_violation(f"{field} does not target an active order")
                pending = orders[order_id]
                if event == "LIMIT_REJECTED_INSUFFICIENT_MARGIN":
                    if payload.get("pair") != pending["pair"]:
                        _mtm_violation(f"{field}.pair does not match its order")
                elif event == "ORDER_CANCEL_CONCURRENCY_CAP":
                    if (
                        payload.get("pair") != pending["pair"]
                        or payload.get("side") != pending["side"]
                        or _number(
                            payload.get("units"),
                            field=f"{field}.units",
                            positive=True,
                        )
                        != pending["units"]
                    ):
                        _mtm_violation(f"{field} does not match its order")
                    validate_event_quote(payload, str(pending["pair"]), field=field)
                del orders[order_id]
            elif event == "SET_EXIT":
                trade_id = _identity(payload.get("trade_id"), field=f"{field}.trade_id")
                if trade_id not in positions:
                    _mtm_violation(f"{field} does not target an active trade")
                positions[trade_id]["tp_price"] = _mtm_optional_number(
                    payload.get("tp"), field=f"{field}.tp"
                )
                positions[trade_id]["sl_price"] = _mtm_optional_number(
                    payload.get("sl"), field=f"{field}.sl"
                )
            elif event in {
                "ORDER_REJECTED_INSUFFICIENT_MARGIN",
                "ORDER_REJECTED_CONCURRENCY_CAP",
            }:
                pair = _pair(payload.get("pair"), field=f"{field}.pair")
                if pair not in trade_pairs:
                    _mtm_violation(f"{field}.pair is not a sealed trade pair")
                side = payload.get("side")
                if side not in {"LONG", "SHORT"}:
                    _mtm_violation(f"{field}.side is invalid")
                proposed_entry = {
                    "pair": pair,
                    "side": side,
                    "units": _number(
                        payload.get("units"), field=f"{field}.units", positive=True
                    ),
                }
                admission = expected_concurrency_admission(
                    proposed_entry, field=f"{field}.admission"
                )
                if event == "ORDER_REJECTED_CONCURRENCY_CAP":
                    if admission is None or payload.get("admission") != admission:
                        _mtm_violation(
                            f"{field} concurrency rejection is not reconstructed"
                        )
                elif admission is not None or margin_headroom_ok(
                    proposed_entry, field=f"{field}.margin_admission"
                ):
                    _mtm_violation(f"{field} margin rejection is not reconstructed")
            else:
                _mtm_violation(f"unexpected stateful event {event}")

        mark_keys = {
            "contract",
            "mark_index",
            "kind",
            "coordinate",
            "batch_index",
            "batch_sha256",
            "feed_cursor",
            "account",
            "account_sha256",
            "positions",
            "positions_sha256",
            "orders",
            "orders_sha256",
            "quotes",
            "quotes_sha256",
            "previous_mark_sha256",
            "mark_sha256",
        }

        def validate_feed_cursor(
            value: Any,
            *,
            expected_coordinate: Mapping[str, Any],
            completed: bool,
            field: str,
        ) -> dict[str, Any]:
            cursor = dict(
                _require_exact_keys(
                    value,
                    field=field,
                    expected={
                        "mode",
                        "epoch",
                        "phase",
                        "bar_count",
                        "completed",
                        "replay_identity_sha256",
                    },
                )
            )
            epoch = int(expected_coordinate["epoch"])
            if latest_epoch != epoch:
                _mtm_violation(f"{field} is not bound to the latest replay epoch")
            epoch_ordinal = epoch_count - 1
            if (
                cursor["mode"] != "replay"
                or _integer(cursor["epoch"], field=f"{field}.epoch") != epoch
                or cursor["phase"] != expected_coordinate["phase"]
                or _integer(
                    cursor["bar_count"], field=f"{field}.bar_count", non_negative=True
                )
                != epoch_ordinal
                or cursor["completed"] is not completed
                or _sha256(
                    cursor["replay_identity_sha256"],
                    field=f"{field}.replay_identity_sha256",
                )
                != replay_identity_sha
            ):
                _mtm_violation(f"{field} is not bound to the replay coordinate")
            return cursor

        def validate_mark(
            payload: Mapping[str, Any],
            *,
            kind: str,
            linked_batch: Mapping[str, Any] | None,
            field: str,
        ) -> Mapping[str, Any]:
            nonlocal previous_mark_sha, mark_count, mtm_equity_peak, mtm_drawdown
            nonlocal peak_margin_jpy, peak_margin_fraction
            mark = dict(_require_exact_keys(payload, field=field, expected=mark_keys))
            if mark["contract"] != _ACCOUNT_MARK_CONTRACT or mark["kind"] != kind:
                _mtm_violation(f"{field} contract/kind mismatch")
            if (
                _integer(
                    mark["mark_index"], field=f"{field}.mark_index", non_negative=True
                )
                != mark_count
            ):
                _mtm_violation(f"{field}.mark_index is not contiguous")
            if mark["previous_mark_sha256"] != previous_mark_sha:
                _mtm_violation(f"{field}.previous_mark_sha256 mismatch")
            claimed_mark_sha = _sha256(
                mark["mark_sha256"], field=f"{field}.mark_sha256"
            )
            if claimed_mark_sha != _canonical_sha256(
                {key: value for key, value in mark.items() if key != "mark_sha256"}
            ):
                _mtm_violation(f"{field}.mark_sha256 mismatch")
            for state_name, hash_name in (
                ("account", "account_sha256"),
                ("positions", "positions_sha256"),
                ("orders", "orders_sha256"),
                ("quotes", "quotes_sha256"),
            ):
                if _sha256(
                    mark[hash_name], field=f"{field}.{hash_name}"
                ) != _canonical_sha256(mark[state_name]):
                    _mtm_violation(f"{field}.{hash_name} mismatch")
            expected_positions = [positions[key] for key in sorted(positions)]
            expected_orders = [orders[key] for key in sorted(orders)]
            expected_quotes = [quotes[key] for key in sorted(quotes)]
            if mark["positions"] != expected_positions:
                _mtm_violation(f"{field}.positions do not match reconstructed state")
            if mark["orders"] != expected_orders:
                _mtm_violation(f"{field}.orders do not match reconstructed state")
            if mark["quotes"] != expected_quotes:
                _mtm_violation(f"{field}.quotes do not match reconstructed state")
            account = mark["account"]
            account_keys = {
                "balance_jpy",
                "equity_jpy",
                "margin_used_jpy",
                "margin_usage",
                "accrued_financing_jpy",
                "open_positions",
                "resting_orders",
            }
            account = dict(
                _require_exact_keys(
                    account, field=f"{field}.account", expected=account_keys
                )
            )
            expected = expected_account(field=f"{field}.account")
            rounding_tolerance = max(0.01, independently_resolved_exits * 0.01)
            for name in ("balance_jpy", "equity_jpy"):
                if not math.isclose(
                    _number(account[name], field=f"{field}.account.{name}"),
                    float(expected[name]),
                    rel_tol=0,
                    abs_tol=rounding_tolerance,
                ):
                    _mtm_violation(f"{field}.account.{name} is not reconstructed")
            for name in ("margin_used_jpy", "accrued_financing_jpy"):
                if (
                    _number(
                        account[name],
                        field=f"{field}.account.{name}",
                        non_negative=True,
                    )
                    != expected[name]
                ):
                    _mtm_violation(f"{field}.account.{name} is not reconstructed")
            if (
                _number(
                    account["margin_usage"],
                    field=f"{field}.account.margin_usage",
                    non_negative=True,
                )
                != expected["margin_usage"]
            ):
                _mtm_violation(f"{field}.account.margin_usage is not reconstructed")
            if positions and float(expected["margin_usage"]) >= 1.0:
                _mtm_violation(f"{field}.account retains exposure past margin closeout")
            for name in ("open_positions", "resting_orders"):
                if (
                    _integer(
                        account[name],
                        field=f"{field}.account.{name}",
                        non_negative=True,
                    )
                    != expected[name]
                ):
                    _mtm_violation(f"{field}.account.{name} is not reconstructed")
            if kind == "START":
                if any(
                    mark[name] is not None
                    for name in (
                        "coordinate",
                        "batch_index",
                        "batch_sha256",
                        "feed_cursor",
                    )
                ):
                    _mtm_violation("START mark must not claim a replay coordinate")
                if (
                    positions
                    or orders
                    or quotes
                    or account
                    != {
                        "balance_jpy": round(start_balance, 2),
                        "equity_jpy": round(start_balance, 2),
                        "margin_used_jpy": 0.0,
                        "margin_usage": 0.0,
                        "accrued_financing_jpy": 0.0,
                        "open_positions": 0,
                        "resting_orders": 0,
                    }
                ):
                    _mtm_violation("START mark is not a pristine initial account")
            elif kind == "PHASE":
                if linked_batch is None:
                    _mtm_violation("PHASE mark has no preceding quote batch")
                if (
                    mark["coordinate"] != linked_batch["coordinate"]
                    or mark["batch_index"] != linked_batch["batch_index"]
                    or mark["batch_sha256"] != linked_batch["batch_sha256"]
                ):
                    _mtm_violation("PHASE mark is not bound to its quote batch")
                validate_feed_cursor(
                    mark["feed_cursor"],
                    expected_coordinate=linked_batch["coordinate"],
                    completed=False,
                    field=f"{field}.feed_cursor",
                )
            else:
                if mark["coordinate"] is not None:
                    _mtm_violation("TERMINAL mark coordinate must be null")
                if linked_batch is None:
                    _mtm_violation("TERMINAL mark has no terminal quote batch")
                if (
                    mark["batch_index"] != linked_batch["batch_index"]
                    or mark["batch_sha256"] != linked_batch["batch_sha256"]
                ):
                    _mtm_violation("TERMINAL mark is not bound to the terminal batch")
                validate_feed_cursor(
                    mark["feed_cursor"],
                    expected_coordinate=linked_batch["coordinate"],
                    completed=True,
                    field=f"{field}.feed_cursor",
                )
            equity = float(expected["equity_jpy"])
            margin = float(expected["margin_used_jpy"])
            mtm_equity_peak = max(mtm_equity_peak, equity)
            if mtm_equity_peak > 0:
                mtm_drawdown = max(
                    mtm_drawdown, (mtm_equity_peak - equity) / mtm_equity_peak
                )
            peak_margin_jpy = max(peak_margin_jpy, margin)
            if equity > 0:
                peak_margin_fraction = max(peak_margin_fraction, margin / equity)
            previous_mark_sha = claimed_mark_sha
            mark_count += 1
            return mark

        record_iterator = iter(records)
        try:
            session_start_record = next(record_iterator)
            bot_loaded_record = next(record_iterator)
            start_mark_record = next(record_iterator)
        except StopIteration:
            _mtm_violation("continuous MTM ledger is truncated before START mark")
        if session_start_record["event"] != "SESSION_START":
            _mtm_violation("continuous MTM ledger does not begin with SESSION_START")
        # A bot constructor must not mutate state and then bless it with START.
        if bot_loaded_record["event"] != "BOT_LOADED":
            _mtm_violation("BOT_LOADED must immediately follow SESSION_START")
        if start_mark_record["event"] != "ACCOUNT_MARK":
            _mtm_violation("START ACCOUNT_MARK must immediately follow BOT_LOADED")
        validate_mark(
            start_mark_record["payload"],
            kind="START",
            linked_batch=None,
            field="ledger[3]",
        )

        batch_keys = {
            "contract",
            "batch_index",
            "coordinate",
            "feed_pairs",
            "batch_pairs",
            "coverage_complete",
            "quotes",
            "quotes_sha256",
            "previous_batch_sha256",
            "batch_sha256",
        }
        stateful_events = (
            _FILL_EVENTS
            | _EXIT_EVENTS
            | _MARGIN_REJECTION_EVENTS
            | {
                "ORDER_LIMIT",
                "ORDER_STOP",
                "ORDER_CANCEL",
                "ORDER_CANCEL_CONCURRENCY_CAP",
                "ORDER_REJECTED_CONCURRENCY_CAP",
                "SET_EXIT",
            }
        )
        settlement_seen = False
        terminal_seen = False
        pre_settlement_orders: list[str] | None = None
        pre_settlement_trades: list[str] | None = None
        latest_batch: Mapping[str, Any] | None = None
        stop: Mapping[str, Any] | None = None
        for record_index, record in enumerate(record_iterator, start=4):
            event = record["event"]
            payload = record["payload"]
            field = f"ledger[{record_index}]"
            if event == "SESSION_STOP":
                stop = payload
                try:
                    next(record_iterator)
                except StopIteration:
                    break
                _mtm_violation("SESSION_STOP is not the final ledger record")
            if event == "QUOTE_BATCH_BEGIN":
                if current_batch is not None or settlement_seen or terminal_seen:
                    _mtm_violation(f"{field} quote batch is out of sequence")
                batch = dict(
                    _require_exact_keys(payload, field=field, expected=batch_keys)
                )
                if batch["contract"] != _QUOTE_BATCH_CONTRACT:
                    _mtm_violation(f"{field} quote batch contract mismatch")
                if (
                    _integer(
                        batch["batch_index"],
                        field=f"{field}.batch_index",
                        non_negative=True,
                    )
                    != batch_count
                ):
                    _mtm_violation(f"{field}.batch_index is not contiguous")
                batch_coordinate = coordinate(
                    batch["coordinate"], field=f"{field}.coordinate"
                )
                phase_offset = batch_count % len(expected_phase_order)
                if batch_coordinate["phase"] != expected_phase_order[phase_offset]:
                    _mtm_violation(f"{field}.coordinate phase order is invalid")
                if phase_offset == 0:
                    if (
                        latest_epoch is not None
                        and batch_coordinate["epoch"] <= latest_epoch
                    ):
                        _mtm_violation(f"{field}.coordinate epoch is not increasing")
                    latest_epoch = int(batch_coordinate["epoch"])
                    epoch_count += 1
                elif batch_coordinate["epoch"] != latest_epoch:
                    _mtm_violation(f"{field}.coordinate skips an intrabar phase")
                batch_quotes = batch["quotes"]
                if not isinstance(batch_quotes, list):
                    _mtm_violation(f"{field}.quotes must be a list")
                batch_pairs = batch["batch_pairs"]
                if batch["feed_pairs"] != list(feed_pairs):
                    _mtm_violation(f"{field} feed_pairs drifted")
                if sparse_union:
                    if (
                        not isinstance(batch_pairs, list)
                        or not batch_pairs
                        or batch_pairs != sorted(set(batch_pairs))
                        or any(pair not in feed_pairs for pair in batch_pairs)
                        or len(batch_quotes) != len(batch_pairs)
                        or not isinstance(batch["coverage_complete"], bool)
                        or batch["coverage_complete"]
                        != (batch_pairs == list(feed_pairs))
                    ):
                        _mtm_violation(f"{field} sparse batch presence is invalid")
                    if phase_offset == 0:
                        current_epoch_batch_pairs = list(batch_pairs)
                        observed_availability_mask.append(
                            {
                                "epoch": int(batch_coordinate["epoch"]),
                                "batch_pairs": list(batch_pairs),
                            }
                        )
                        if batch_pairs == list(feed_pairs):
                            observed_full_epoch_count += 1
                        else:
                            observed_partial_epoch_count += 1
                        for pair in batch_pairs:
                            observed_pair_row_counts[pair] += 1
                    elif batch_pairs != current_epoch_batch_pairs:
                        _mtm_violation(
                            f"{field} sparse availability changes within one bar"
                        )
                elif (
                    batch_pairs != list(feed_pairs)
                    or batch["coverage_complete"] is not True
                    or len(batch_quotes) != len(feed_pairs)
                ):
                    _mtm_violation(f"{field} does not cover every feed pair exactly")
                normalized_batch_quotes: list[dict[str, Any]] = []
                watermark = quote_sequence + len(batch_quotes)
                for pair_index, (expected_pair, raw_quote) in enumerate(
                    zip(batch_pairs, batch_quotes, strict=True), start=1
                ):
                    quote = dict(
                        _require_exact_keys(
                            raw_quote,
                            field=f"{field}.quotes[{pair_index - 1}]",
                            expected={"pair", "bid", "ask", "ts"},
                        )
                    )
                    pair = _pair(quote["pair"], field=f"{field}.quotes.pair")
                    bid = _number(
                        quote["bid"], field=f"{field}.quotes.bid", positive=True
                    )
                    ask = _number(
                        quote["ask"], field=f"{field}.quotes.ask", positive=True
                    )
                    if pair != expected_pair or bid > ask:
                        _mtm_violation(
                            f"{field}.quotes are not canonical executable quotes"
                        )
                    timestamp = quote["ts"]
                    if not isinstance(timestamp, str) or not timestamp.endswith(
                        f"#{batch_coordinate['phase']}"
                    ):
                        _mtm_violation(f"{field}.quote timestamp phase mismatch")
                    parsed_timestamp = _parse_utc(
                        timestamp, field=f"{field}.quotes.ts", allow_naive=True
                    )
                    if parsed_timestamp.timestamp() != batch_coordinate["epoch"]:
                        _mtm_violation(f"{field}.quote timestamp epoch mismatch")
                    quote_sequence += 1
                    normalized = {"pair": pair, "bid": bid, "ask": ask, "ts": timestamp}
                    normalized_batch_quotes.append(normalized)
                    quotes[pair] = {
                        **normalized,
                        "sequence": quote_sequence,
                        "watermark": watermark,
                    }
                    history = quote_history.setdefault(pair, [])
                    history.append(dict(quotes[pair]))
                    if len(history) > 128:
                        del history[:-128]
                if batch_quotes != normalized_batch_quotes:
                    _mtm_violation(f"{field}.quotes use noncanonical numeric state")
                if corpus_batches is not None:
                    try:
                        corpus_coordinate, corpus_quotes = next(corpus_batches)
                    except StopIteration:
                        _mtm_violation(
                            f"{field} has no corresponding sealed corpus coordinate"
                        )
                    if (
                        batch_coordinate != corpus_coordinate
                        or normalized_batch_quotes != corpus_quotes
                    ):
                        _mtm_violation(
                            f"{field} quote batch does not match sealed corpus bytes"
                        )
                if _sha256(
                    batch["quotes_sha256"], field=f"{field}.quotes_sha256"
                ) != _canonical_sha256(batch_quotes):
                    _mtm_violation(f"{field}.quotes_sha256 mismatch")
                if batch["previous_batch_sha256"] != previous_batch_sha:
                    _mtm_violation(f"{field}.previous_batch_sha256 mismatch")
                batch_sha = _sha256(
                    batch["batch_sha256"], field=f"{field}.batch_sha256"
                )
                if batch_sha != _canonical_sha256(
                    {
                        key: value
                        for key, value in batch.items()
                        if key != "batch_sha256"
                    }
                ):
                    _mtm_violation(f"{field}.batch_sha256 mismatch")
                previous_batch_sha = batch_sha
                batch["coordinate"] = batch_coordinate
                current_batch = batch
                current_batch_manual_started = False
                current_market_consequence_trade_id = None
                current_market_consequence_window = False
                current_order_consequence_order_id = None
                current_order_consequence_window = False
                margin_closeout_sequence_active = False
                batch_pair_set = set(batch_pairs)
                batch_eligible_order_ids = {
                    order_id
                    for order_id, order in orders.items()
                    if order["pair"] in batch_pair_set
                }
                batch_eligible_position_ids = {
                    trade_id
                    for trade_id, position in positions.items()
                    if position["pair"] in batch_pair_set
                }
                latest_batch = batch
                if first_observed_coordinate is None:
                    first_observed_coordinate = batch_coordinate
                last_observed_coordinate = batch_coordinate
                batch_count += 1
            elif event == "ACCOUNT_MARK":
                kind = payload.get("kind")
                if kind == "PHASE":
                    if current_batch is None:
                        _mtm_violation(f"{field} PHASE mark has no open batch")
                    require_no_mandatory_event(field=f"{field}.pre_phase")
                    validate_mark(
                        payload,
                        kind="PHASE",
                        linked_batch=current_batch,
                        field=field,
                    )
                    current_batch = None
                    if batch_count == expected_batch_count:
                        pre_settlement_orders = sorted(orders)
                        pre_settlement_trades = sorted(positions)
                elif kind == "TERMINAL":
                    if (
                        not settlement_seen
                        or terminal_seen
                        or current_batch is not None
                    ):
                        _mtm_violation(f"{field} TERMINAL mark is out of sequence")
                    terminal_mark_payload = validate_mark(
                        payload,
                        kind="TERMINAL",
                        linked_batch=latest_batch,
                        field=field,
                    )
                    terminal_seen = True
                else:
                    _mtm_violation(f"{field} contains an unexpected ACCOUNT_MARK kind")
            elif event in stateful_events:
                if terminal_seen or settlement_seen:
                    _mtm_violation(f"{field} mutates state after settlement")
                if current_batch is None:
                    if (
                        batch_count != expected_batch_count
                        or pre_settlement_orders is None
                    ):
                        _mtm_violation(f"{field} mutates state outside a quote batch")
                    if event not in {"ORDER_CANCEL", "CLOSE"}:
                        _mtm_violation(f"{field} is not a permitted settlement action")
                    if event == "CLOSE" and sparse_union:
                        position = positions.get(str(payload.get("trade_id")))
                        if position is None or latest_batch is None:
                            _mtm_violation(
                                f"{field} settlement close has no terminal position/batch"
                            )
                        pair = str(position["pair"])
                        terminal_coordinate = latest_batch["coordinate"]
                        quote = current_quote(pair, field=field)
                        quote_epoch = int(
                            _parse_utc(
                                quote["ts"],
                                field=f"{field}.quote.ts",
                                allow_naive=True,
                            ).timestamp()
                        )
                        if (
                            terminal_coordinate["phase"] != "C"
                            or pair not in latest_batch["batch_pairs"]
                            or quote_epoch != int(terminal_coordinate["epoch"])
                            or not str(quote["ts"]).endswith("#C")
                        ):
                            _mtm_violation(
                                f"{field} settlement close lacks a fresh terminal quote"
                            )
                else:
                    bot_origin_events = {
                        "ORDER_LIMIT",
                        "ORDER_STOP",
                        "ORDER_CANCEL",
                        "FILL_MARKET",
                        "SET_EXIT",
                        "CLOSE",
                        "ORDER_REJECTED_INSUFFICIENT_MARGIN",
                        "ORDER_REJECTED_CONCURRENCY_CAP",
                    }
                    if event in bot_origin_events:
                        if (
                            current_batch["coordinate"]["phase"] != "O"
                            or epoch_count <= 1
                        ):
                            _mtm_violation(
                                f"{field} bot-origin action is outside a causal O callback"
                            )
                        if sparse_union:
                            if set(quotes) != set(feed_pairs):
                                _mtm_violation(
                                    f"{field} bot-origin action precedes feed priming"
                                )
                            if event in {
                                "ORDER_LIMIT",
                                "ORDER_STOP",
                                "FILL_MARKET",
                                "ORDER_REJECTED_INSUFFICIENT_MARGIN",
                                "ORDER_REJECTED_CONCURRENCY_CAP",
                            }:
                                target_pair = payload.get("pair")
                            elif event == "ORDER_CANCEL":
                                target = orders.get(str(payload.get("order_id")))
                                target_pair = target.get("pair") if target else None
                            else:
                                target = positions.get(str(payload.get("trade_id")))
                                target_pair = target.get("pair") if target else None
                            if target_pair not in current_batch["batch_pairs"]:
                                _mtm_violation(
                                    f"{field} bot-origin action targets an absent pair"
                                )
                        require_no_mandatory_event(field=f"{field}.pre_bot_action")
                        batch_eligible_order_ids.clear()
                        batch_eligible_position_ids.clear()
                        current_batch_manual_started = True
                        current_market_consequence_trade_id = None
                        current_market_consequence_window = False
                        current_order_consequence_order_id = None
                        current_order_consequence_window = False
                    elif current_batch_manual_started:
                        trade_id = payload.get("trade_id")
                        order_id = payload.get("order_id")
                        immediate_order_consequence = (
                            event
                            in {
                                "FILL_LIMIT",
                                "LIMIT_REJECTED_INSUFFICIENT_MARGIN",
                                "ORDER_CANCEL_CONCURRENCY_CAP",
                            }
                            and current_order_consequence_window
                            and order_id == current_order_consequence_order_id
                        )
                        immediate_market_exit = (
                            event in {"EXIT_TP", "EXIT_SL"}
                            and trade_id == current_market_consequence_trade_id
                        )
                        immediate_market_closeout = (
                            event == "MARGIN_CLOSEOUT"
                            and current_market_consequence_window
                        )
                        if (
                            not immediate_order_consequence
                            and not immediate_market_exit
                            and not immediate_market_closeout
                        ):
                            _mtm_violation(
                                f"{field} asynchronous broker event follows an O action"
                            )
                        require_mandatory_event_matches(event, payload, field=field)
                    else:
                        require_mandatory_event_matches(event, payload, field=field)
                    if (
                        margin_closeout_sequence_active
                        and event != "MARGIN_CLOSEOUT"
                        and event not in bot_origin_events
                    ):
                        _mtm_violation(
                            f"{field} interrupts an atomic margin closeout sequence"
                        )
                    if event == "MARGIN_CLOSEOUT":
                        if not margin_closeout_sequence_active:
                            pre_close_account = expected_account(
                                field=f"{field}.pre_margin_closeout"
                            )
                            if float(pre_close_account["margin_usage"]) < 1.0:
                                _mtm_violation(
                                    f"{field} margin closeout has no margin trigger"
                                )
                            margin_closeout_sequence_active = True
                    else:
                        margin_closeout_sequence_active = False
                apply_action(event, payload, field=field)
                if current_batch is not None:
                    if event in {"ORDER_LIMIT", "ORDER_STOP"}:
                        order_id = str(payload.get("order_id"))
                        if touched_order(
                            orders[order_id], field=f"{field}.submission_marketability"
                        ):
                            batch_eligible_order_ids.add(order_id)
                            current_order_consequence_order_id = order_id
                            current_order_consequence_window = True
                    elif event == "FILL_LIMIT":
                        batch_eligible_order_ids.discard(str(payload.get("order_id")))
                        current_order_consequence_window = False
                        current_market_consequence_trade_id = str(
                            payload.get("trade_id")
                        )
                        current_market_consequence_window = True
                        batch_eligible_position_ids.add(
                            current_market_consequence_trade_id
                        )
                    elif event in {
                        "LIMIT_REJECTED_INSUFFICIENT_MARGIN",
                        "ORDER_CANCEL_CONCURRENCY_CAP",
                    }:
                        batch_eligible_order_ids.discard(str(payload.get("order_id")))
                        current_order_consequence_window = False
                    elif event == "SET_EXIT":
                        trade_id = str(payload.get("trade_id"))
                        if touched_exit(
                            positions[trade_id],
                            field=f"{field}.submission_exit_marketability",
                        ):
                            current_market_consequence_trade_id = trade_id
                            current_market_consequence_window = True
                            batch_eligible_position_ids.add(trade_id)
                    elif event in {"EXIT_TP", "EXIT_SL", "MARGIN_CLOSEOUT"}:
                        batch_eligible_position_ids.discard(
                            str(payload.get("trade_id"))
                        )
                    if event == "FILL_MARKET":
                        current_market_consequence_trade_id = str(
                            payload.get("trade_id")
                        )
                        current_market_consequence_window = True
                        batch_eligible_position_ids.add(
                            current_market_consequence_trade_id
                        )
            elif event == "PERIOD_END_SETTLEMENT":
                if (
                    settlement_seen
                    or terminal_seen
                    or current_batch is not None
                    or batch_count != expected_batch_count
                    or pre_settlement_orders is None
                    or pre_settlement_trades is None
                ):
                    _mtm_violation(f"{field} settlement is out of sequence")
                requested_orders = payload.get("requested_order_ids")
                requested_trades = payload.get("requested_trade_ids")
                if (
                    not isinstance(requested_orders, list)
                    or not isinstance(requested_trades, list)
                    or sorted(requested_orders) != pre_settlement_orders
                    or sorted(requested_trades) != pre_settlement_trades
                    or len(set(requested_orders)) != len(requested_orders)
                    or len(set(requested_trades)) != len(requested_trades)
                ):
                    _mtm_violation(
                        f"{field} settlement targets do not match phase state"
                    )
                if positions or orders:
                    _mtm_violation(
                        f"{field} claims complete settlement with live state"
                    )
                settlement_seen = True
            else:
                _mtm_violation(f"{field} unexpected event {event} in MTM contract")

        if current_batch is not None:
            _mtm_violation("terminal quote batch has no PHASE mark")
        if batch_count != expected_batch_count:
            _mtm_violation("observed quote batch count does not match the manifest")
        if sparse_union:
            if (
                epoch_count != expected_union_epoch_count
                or observed_full_epoch_count != expected_full_epoch_count
                or observed_partial_epoch_count != expected_partial_epoch_count
                or observed_pair_row_counts != expected_pair_row_counts
                or quote_sequence != expected_quote_count
                or _canonical_sha256(observed_availability_mask)
                != expected_availability_mask_sha
            ):
                _mtm_violation(
                    "observed sparse availability/counts do not match the manifest"
                )
        if first_observed_coordinate != expected_first_coordinate:
            _mtm_violation("first coordinate does not match the manifest")
        if last_observed_coordinate != expected_last_coordinate:
            _mtm_violation("last coordinate does not match the manifest")
        if corpus_batches is not None:
            try:
                next(corpus_batches)
            except StopIteration:
                pass
            else:
                _mtm_violation("sealed corpus has unconsumed replay coordinates")
        if previous_batch_sha != expected_batch_terminal:
            _mtm_violation("terminal quote-batch chain does not match the manifest")
        if cached_corpus_terminal is None:
            _CORPUS_COMMITMENT_CACHE[corpus_cache_key] = previous_batch_sha
            _CORPUS_COMMITMENT_CACHE.move_to_end(corpus_cache_key)
            while len(_CORPUS_COMMITMENT_CACHE) > _MAX_CORPUS_COMMITMENT_CACHE:
                _CORPUS_COMMITMENT_CACHE.popitem(last=False)
        if not settlement_seen or not terminal_seen or terminal_mark_payload is None:
            _mtm_violation("settlement and terminal mark are not complete")
        if mark_count != expected_batch_count + 2:
            _mtm_violation("START/PHASE/TERMINAL mark count is incomplete")
        if stop is None:
            _mtm_violation("continuous MTM ledger has no SESSION_STOP")
        if (
            stop.get("mtm_complete") is not True
            or stop.get("mtm_mark_count") != mark_count
            or stop.get("mtm_terminal_mark_sha256") != previous_mark_sha
            or stop.get("account_error") is not None
            or stop.get("account") != terminal_mark_payload["account"]
        ):
            _mtm_violation("SESSION_STOP does not bind the verified terminal mark")
        return {
            "mtm_complete": True,
            "mtm_evidence_status": "VERIFIED_COORDINATE_COMPLETE_ACCOUNT_MARK_CHAIN",
            "mtm_mark_count": mark_count,
            "mtm_max_drawdown_fraction": mtm_drawdown,
            "peak_margin_jpy": peak_margin_jpy,
            "peak_margin_fraction": peak_margin_fraction,
            "terminal_balance_jpy": reconstructed_balance,
        }
    except DojoBotTrainerError as exc:
        if str(exc).startswith("MTM_CONTRACT_VIOLATION:"):
            raise
        _mtm_violation(str(exc))


def _iter_verified_ledger_records(
    handle: BinaryIO,
    *,
    expected_owner: str,
    artifact_stats: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Replay a ledger with bounded memory from a seekable authenticated stream."""

    try:
        handle.seek(0)
    except (AttributeError, OSError) as exc:
        raise DojoBotTrainerError(
            "ledger stream must be seekable for bounded multi-pass verification"
        ) from exc
    previous = _ZERO_SHA256
    ledger_size = 0
    record_count = 0
    file_digest = hashlib.sha256() if artifact_stats is not None else None
    for line_number, raw_line in enumerate(handle, start=1):
        if not isinstance(raw_line, (bytes, bytearray)):
            raise DojoBotTrainerError("ledger stream must yield binary rows")
        raw_bytes = bytes(raw_line)
        ledger_size += len(raw_bytes)
        if ledger_size > _MAX_LEDGER_BYTES:
            raise DojoBotTrainerError("ledger artifact exceeds the size limit")
        if file_digest is not None:
            file_digest.update(raw_bytes)
        if not raw_bytes.strip():
            raise DojoBotTrainerError(f"blank ledger row at line {line_number}")
        if len(raw_bytes) > MAX_JSON_BYTES:
            raise DojoBotTrainerError("ledger row exceeds the JSON size limit")
        row = dict(
            _require_exact_keys(
                _load_jsonish(raw_bytes, field=f"ledger[{line_number}]"),
                field=f"ledger[{line_number}]",
                expected={"ts_utc", "event", "payload", "prev_sha", "sha"},
            )
        )
        _parse_utc(row["ts_utc"], field=f"ledger[{line_number}].ts_utc")
        if not isinstance(row["event"], str) or not isinstance(row["payload"], Mapping):
            raise DojoBotTrainerError("ledger event/payload is malformed")
        if row["prev_sha"] != previous:
            raise DojoBotTrainerError("ledger hash chain predecessor mismatch")
        claimed = _sha256(row["sha"], field=f"ledger[{line_number}].sha")
        body = {key: item for key, item in row.items() if key != "sha"}
        if claimed != _canonical_sha256(body):
            raise DojoBotTrainerError("ledger row digest mismatch")
        previous = claimed
        if (
            row["event"] in _OWNED_LEDGER_EVENTS
            and row["payload"].get("strategy_owner_id") != expected_owner
        ):
            raise DojoBotTrainerError(
                "owned ledger event has missing or mismatched strategy owner"
            )
        record_count += 1
        yield row
    if artifact_stats is not None:
        artifact_stats.update(
            {
                "ledger_size": ledger_size,
                "ledger_file_sha256": file_digest.hexdigest(),
                "ledger_terminal_sha256": previous,
                "ledger_record_count": record_count,
            }
        )


def score_ledger_metrics(
    ledger_path: Path | BinaryIO,
    start_balance_jpy: float,
    expected_pairs: Sequence[str],
    window_start: str,
    window_end: str,
    *,
    expected_intrabar: str,
    expected_slippage_pips_per_fill: float,
    expected_financing_pips_per_day: float,
    expected_corpus_sha256: str,
    expected_bot_config_sha256: str,
    expected_strategy_owner_id: str,
    expected_bot_module_sha256: str,
    expected_bot_dependency_sha256: Mapping[str, str],
    expected_feed_pairs: Sequence[str] | None = None,
    ledger_artifact_path: str | None = None,
    expected_max_concurrent_per_pair: int | None = None,
    expected_global_max_concurrent: int | None = None,
) -> dict[str, Any]:
    """Rebuild deterministic TRAIN diagnostics from a broker hash-chain ledger.

    Legacy ledgers do not contain a complete account mark at every atomic quote
    batch.  Such ledgers remain useful for realized and capital-lock diagnostics,
    but this function marks ``mtm_complete=false`` and never treats them as
    promotion evidence.  Additive leave-one-pair-out attribution is reported
    separately from a true counterfactual LOPO replay.
    """

    start_balance = _number(start_balance_jpy, field="start_balance_jpy", positive=True)
    pairs = sorted(
        _pair(pair, field=f"expected_pairs[{index}]")
        for index, pair in enumerate(expected_pairs)
    )
    if not pairs or len(set(pairs)) != len(pairs):
        raise DojoBotTrainerError("expected_pairs must be non-empty and unique")
    raw_feed_pairs = (
        expected_pairs if expected_feed_pairs is None else expected_feed_pairs
    )
    feed_pairs = sorted(
        _pair(pair, field=f"expected_feed_pairs[{index}]")
        for index, pair in enumerate(raw_feed_pairs)
    )
    if not feed_pairs or len(set(feed_pairs)) != len(feed_pairs):
        raise DojoBotTrainerError("expected_feed_pairs must be non-empty and unique")
    if not set(pairs).issubset(feed_pairs):
        raise DojoBotTrainerError("expected_feed_pairs must include expected_pairs")
    start = _parse_utc(window_start, field="window_start", allow_naive=True)
    end = _parse_utc(window_end, field="window_end", allow_naive=True)
    if end <= start:
        raise DojoBotTrainerError("ledger scoring window is empty")
    if expected_intrabar not in REQUIRED_INTRABAR_PATHS:
        raise DojoBotTrainerError("expected_intrabar must be OHLC or OLHC")
    expected_slippage = _number(
        expected_slippage_pips_per_fill,
        field="expected_slippage_pips_per_fill",
        non_negative=True,
    )
    expected_financing = _number(
        expected_financing_pips_per_day,
        field="expected_financing_pips_per_day",
        non_negative=True,
    )
    expected_corpus_sha = _sha256(
        expected_corpus_sha256, field="expected_corpus_sha256"
    )
    expected_config_sha = _sha256(
        expected_bot_config_sha256, field="expected_bot_config_sha256"
    )
    expected_owner = _identity(
        expected_strategy_owner_id, field="expected_strategy_owner_id"
    )
    expected_module_sha = _sha256(
        expected_bot_module_sha256, field="expected_bot_module_sha256"
    )
    expected_dependencies = _normalize_source_digests(expected_bot_dependency_sha256)
    if (expected_max_concurrent_per_pair is None) != (
        expected_global_max_concurrent is None
    ):
        raise DojoBotTrainerError(
            "expected owner concurrency caps must be supplied together"
        )
    for field, value in (
        ("expected_max_concurrent_per_pair", expected_max_concurrent_per_pair),
        ("expected_global_max_concurrent", expected_global_max_concurrent),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise DojoBotTrainerError(f"{field} must be a positive integer")

    supplied_handle = hasattr(ledger_path, "read")
    if not supplied_handle:
        path = Path(ledger_path)
        try:
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise DojoBotTrainerError("ledger file must be a regular non-symlink")
            flags = (
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(path, flags)
        except DojoBotTrainerError:
            raise
        except OSError as exc:
            raise DojoBotTrainerError("ledger file is missing") from exc
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != (
                info.st_dev,
                info.st_ino,
            ):
                raise DojoBotTrainerError("ledger artifact changed before verification")
            with os.fdopen(descriptor, "rb") as local_handle:
                descriptor = -1
                scored = score_ledger_metrics(
                    local_handle,
                    start_balance_jpy,
                    expected_pairs,
                    window_start,
                    window_end,
                    expected_intrabar=expected_intrabar,
                    expected_slippage_pips_per_fill=expected_slippage_pips_per_fill,
                    expected_financing_pips_per_day=expected_financing_pips_per_day,
                    expected_corpus_sha256=expected_corpus_sha256,
                    expected_bot_config_sha256=expected_bot_config_sha256,
                    expected_strategy_owner_id=expected_strategy_owner_id,
                    expected_bot_module_sha256=expected_bot_module_sha256,
                    expected_bot_dependency_sha256=expected_bot_dependency_sha256,
                    expected_feed_pairs=expected_feed_pairs,
                    ledger_artifact_path=ledger_artifact_path or str(path.resolve()),
                    expected_max_concurrent_per_pair=(expected_max_concurrent_per_pair),
                    expected_global_max_concurrent=expected_global_max_concurrent,
                )
                after = os.fstat(local_handle.fileno())
                if (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                ) != (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                ):
                    raise DojoBotTrainerError(
                        "ledger artifact changed during verification"
                    )
                return scored
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    handle = ledger_path
    if not hasattr(handle, "readline"):
        raise DojoBotTrainerError("ledger stream is not readable")
    ledger_label = ledger_artifact_path or "<verified-ledger-stream>"
    artifact_stats: dict[str, Any] = {}
    first_record: Mapping[str, Any] | None = None
    last_record: Mapping[str, Any] | None = None
    start_count = 0
    stop_count = 0
    loaded: list[Mapping[str, Any]] = []
    settlements: list[Mapping[str, Any]] = []
    for row in _iter_verified_ledger_records(
        handle, expected_owner=expected_owner, artifact_stats=artifact_stats
    ):
        if first_record is None:
            first_record = row
        last_record = row
        if row["event"] == "SESSION_START":
            start_count += 1
        elif row["event"] == "SESSION_STOP":
            stop_count += 1
        elif row["event"] == "BOT_LOADED":
            if len(loaded) < 2:
                loaded.append(row)
        elif row["event"] == "PERIOD_END_SETTLEMENT":
            if len(settlements) < 2:
                settlements.append(row)
    if first_record is None or first_record["event"] != "SESSION_START":
        raise DojoBotTrainerError("ledger must begin with SESSION_START")
    if last_record is None or last_record["event"] != "SESSION_STOP":
        raise DojoBotTrainerError("ledger must end with SESSION_STOP")
    if start_count != 1 or stop_count != 1:
        raise DojoBotTrainerError("ledger session boundaries are not unique")

    start_payload = first_record["payload"]
    declared_pairs = sorted(
        item.strip()
        for item in str(start_payload.get("pairs", "")).split(",")
        if item.strip()
    )
    if declared_pairs != feed_pairs:
        raise DojoBotTrainerError("ledger pair set does not match expected_feed_pairs")
    if not math.isclose(
        _number(
            start_payload.get("balance"), field="SESSION_START.balance", positive=True
        ),
        start_balance,
        rel_tol=0,
        abs_tol=0.01,
    ):
        raise DojoBotTrainerError("ledger start balance mismatch")
    if start_payload.get("order_authority") != "NONE":
        raise DojoBotTrainerError("ledger has unexpected order authority")
    if (
        start_payload.get("contract") != "QR_VIRTUAL_MARKET_SESSION_V1"
        or start_payload.get("feed") != "replay"
    ):
        raise DojoBotTrainerError("ledger is not a replay virtual-market session")
    manifest = start_payload.get("reproducibility_manifest")
    if not isinstance(manifest, Mapping):
        raise DojoBotTrainerError("ledger reproducibility manifest is missing")
    manifest_sha = _sha256(
        manifest.get("manifest_sha256"), field="manifest.manifest_sha256"
    )
    manifest_body = {
        key: item for key, item in manifest.items() if key != "manifest_sha256"
    }
    if manifest_sha != _canonical_sha256(manifest_body):
        raise DojoBotTrainerError("reproducibility manifest hash mismatch")
    start_manifest_sha = _sha256(
        start_payload.get("reproducibility_manifest_sha256"),
        field="SESSION_START.reproducibility_manifest_sha256",
    )
    if start_manifest_sha != manifest_sha:
        raise DojoBotTrainerError("SESSION_START manifest digest mismatch")
    if (
        manifest.get("schema") != "QR_VIRTUAL_SESSION_REPRODUCIBILITY_V1"
        or manifest.get("order_authority") != "NONE"
        or manifest.get("resume_snapshot") is not None
    ):
        raise DojoBotTrainerError("reproducibility manifest safety boundary mismatch")
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise DojoBotTrainerError("manifest source binding is missing")
    repository_root = Path(__file__).resolve().parents[2]
    local_sources = {
        "session_script_sha256": repository_root
        / "scripts"
        / "run-virtual-market-session.py",
        "virtual_broker_sha256": Path(__file__)
        .resolve()
        .with_name("virtual_broker.py"),
    }
    for field, source_path in local_sources.items():
        if not source_path.is_file():
            raise DojoBotTrainerError("bound virtual-market source file is missing")
        expected_source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if (
            _sha256(source.get(field), field=f"manifest.source.{field}")
            != expected_source_sha
        ):
            raise DojoBotTrainerError("virtual-market source digest drift")
    replay = manifest.get("replay")
    costs = manifest.get("costs")
    corpus = manifest.get("corpus")
    bot = manifest.get("bot")
    if (
        not isinstance(replay, Mapping)
        or not isinstance(costs, Mapping)
        or not isinstance(corpus, Mapping)
        or not isinstance(bot, Mapping)
    ):
        raise DojoBotTrainerError("ledger replay/cost/corpus/bot binding is missing")
    replay_start = _parse_utc(
        replay.get("time_from"), field="manifest.replay.time_from", allow_naive=True
    )
    replay_end = _parse_utc(
        replay.get("time_to"), field="manifest.replay.time_to", allow_naive=True
    )
    if replay_start != start or replay_end != end:
        raise DojoBotTrainerError("ledger replay window mismatch")
    if sorted(replay.get("pairs", [])) != feed_pairs:
        raise DojoBotTrainerError("manifest replay pair set mismatch")
    if (
        replay.get("feed") != "replay"
        or replay.get("intrabar") != expected_intrabar
        or replay.get("granularity") != "M1"
        or replay.get("bot_bar") != "feed"
        or replay.get("period_end_settlement") is not True
    ):
        raise DojoBotTrainerError("manifest replay mechanics mismatch")
    manifest_slippage = _number(
        costs.get("slippage_pips_per_fill"),
        field="manifest.costs.slippage_pips_per_fill",
        non_negative=True,
    )
    manifest_financing = _number(
        costs.get("financing_pips_per_day"),
        field="manifest.costs.financing_pips_per_day",
        non_negative=True,
    )
    if (
        manifest_slippage != expected_slippage
        or manifest_financing != expected_financing
    ):
        raise DojoBotTrainerError("manifest costs do not match sealed study costs")
    leverage = _number(
        costs.get("leverage"), field="manifest.costs.leverage", positive=True
    )
    if leverage != _EXPECTED_VIRTUAL_BROKER_LEVERAGE:
        raise DojoBotTrainerError("manifest leverage is not the fixed broker leverage")
    if not math.isclose(
        _number(
            manifest.get("initial_balance_jpy"),
            field="manifest.initial_balance_jpy",
            positive=True,
        ),
        start_balance,
        rel_tol=0,
        abs_tol=0.01,
    ):
        raise DojoBotTrainerError("manifest initial balance mismatch")
    corpus_sha = _sha256(
        corpus.get("corpus_sha256"), field="manifest.corpus.corpus_sha256"
    )
    corpus_body = {key: item for key, item in corpus.items() if key != "corpus_sha256"}
    if corpus_sha != _canonical_sha256(corpus_body):
        raise DojoBotTrainerError("manifest corpus digest mismatch")
    if corpus_sha != expected_corpus_sha:
        raise DojoBotTrainerError("manifest corpus does not match sealed study corpus")
    if (
        bot.get("kind") != "custom_module"
        or bot.get("class") != "Bot"
        or bot.get("strategy_owner_id") != expected_owner
        or _sha256(bot.get("module_sha256"), field="manifest.bot.module_sha256")
        != expected_module_sha
        or bot.get("dependency_sha256") != expected_dependencies
    ):
        raise DojoBotTrainerError("manifest custom bot binding mismatch")
    bindings = bot.get("configuration_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {"DOJO_BOT_CONFIG"}:
        raise DojoBotTrainerError("manifest bot configuration binding mismatch")
    config_binding = _require_exact_keys(
        bindings["DOJO_BOT_CONFIG"],
        field="manifest.bot.configuration_bindings.DOJO_BOT_CONFIG",
        expected={"sha256", "length"},
    )
    if (
        _sha256(
            config_binding.get("sha256"),
            field="manifest.bot.configuration_bindings.DOJO_BOT_CONFIG.sha256",
        )
        != expected_config_sha
    ):
        raise DojoBotTrainerError("manifest bot config digest mismatch")
    config_length = _integer(
        config_binding.get("length"),
        field="manifest.bot.configuration_bindings.DOJO_BOT_CONFIG.length",
        non_negative=True,
    )
    if not 0 < config_length <= MAX_JSON_BYTES:
        raise DojoBotTrainerError("manifest bot config length is unreasonable")
    if len(loaded) != 1:
        raise DojoBotTrainerError("ledger requires exactly one BOT_LOADED receipt")
    loaded_payload = loaded[0]["payload"]
    if (
        loaded_payload.get("strategy_owner_id") != expected_owner
        or loaded_payload.get("class") != "Bot"
        or loaded_payload.get("module") != bot.get("module_path")
    ):
        raise DojoBotTrainerError("BOT_LOADED receipt does not match manifest bot")
    if len(settlements) != 1:
        raise DojoBotTrainerError(
            "ledger requires exactly one PERIOD_END_SETTLEMENT receipt"
        )
    settlement = settlements[0]["payload"]
    if (
        settlement.get("strategy_owner_id") != expected_owner
        or settlement.get("complete") is not True
        or settlement.get("errors") != []
    ):
        raise DojoBotTrainerError("period-end settlement receipt is incomplete")

    verified_mtm = _verify_coordinate_mtm_contract(
        _iter_verified_ledger_records(handle, expected_owner=expected_owner),
        manifest=manifest,
        replay=replay,
        feed_pairs=feed_pairs,
        trade_pairs=pairs,
        start_balance=start_balance,
        expected_owner=expected_owner,
        expected_intrabar=expected_intrabar,
        expected_slippage=expected_slippage,
        expected_financing=expected_financing,
        leverage=leverage,
        expected_max_concurrent_per_pair=expected_max_concurrent_per_pair,
        expected_global_max_concurrent=expected_global_max_concurrent,
    )

    active: dict[str, dict[str, Any]] = {}
    seen_trade_ids: set[str] = set()
    realized_balance = start_balance
    realized_peak = start_balance
    realized_drawdown = 0.0
    pair_pnl = {pair: 0.0 for pair in pairs}
    pair_entries = {pair: 0 for pair in pairs}
    pair_active = {pair: 0 for pair in pairs}
    pair_peak_active = {pair: 0 for pair in pairs}
    fill_count = 0
    resolved_exit_slices = 0
    margin_rejections = 0
    margin_closeouts = 0
    peak_margin_jpy = 0.0
    peak_margin_fraction = 0.0
    capital_lock_jpy_hours = 0.0
    unit_hold_hours = 0.0
    closed_units = 0.0
    account_marks = 0
    mtm_equity_peak = start_balance
    mtm_drawdown = 0.0

    for index, row in enumerate(
        _iter_verified_ledger_records(handle, expected_owner=expected_owner), start=1
    ):
        event = row["event"]
        payload = row["payload"]
        if event in {"SESSION_START", "SESSION_STOP"}:
            continue
        if event in _MARGIN_REJECTION_EVENTS:
            margin_rejections += 1
        if event in _FILL_EVENTS:
            if (
                _number(
                    payload.get("slippage_pips"),
                    field=f"ledger[{index}].slippage_pips",
                    non_negative=True,
                )
                != expected_slippage
            ):
                raise DojoBotTrainerError(
                    "ledger fill slippage does not match sealed study costs"
                )
            trade_id = _identity(
                payload.get("trade_id"), field=f"ledger[{index}].trade_id"
            )
            pair = _pair(payload.get("pair"), field=f"ledger[{index}].pair")
            if pair not in pair_pnl or trade_id in seen_trade_ids:
                raise DojoBotTrainerError("ledger fill pair/trade id is invalid")
            units = _number(
                payload.get("units"), field=f"ledger[{index}].units", positive=True
            )
            entry = _number(
                payload.get("entry", payload.get("price")),
                field=f"ledger[{index}].entry",
                positive=True,
            )
            conversion = payload.get("conversion")
            if not isinstance(conversion, Mapping):
                raise DojoBotTrainerError("ledger fill conversion is missing")
            rate = _number(
                conversion.get("rate_jpy_per_quote_unit"),
                field=f"ledger[{index}].conversion.rate",
                positive=True,
            )
            opened = _sim_quote_time(payload, field=f"ledger[{index}]")
            if opened < start or opened > end:
                raise DojoBotTrainerError("ledger fill is outside the scoring window")
            active[trade_id] = {
                "pair": pair,
                "units": units,
                "opened": opened,
                "margin_per_unit": entry * rate / leverage,
            }
            seen_trade_ids.add(trade_id)
            fill_count += 1
            pair_entries[pair] += 1
            pair_active[pair] += 1
            pair_peak_active[pair] = max(pair_peak_active[pair], pair_active[pair])
        elif event in _EXIT_EVENTS:
            if (
                _number(
                    payload.get("slippage_pips"),
                    field=f"ledger[{index}].slippage_pips",
                    non_negative=True,
                )
                != expected_slippage
            ):
                raise DojoBotTrainerError(
                    "ledger exit slippage does not match sealed study costs"
                )
            trade_id = _identity(
                payload.get("trade_id"), field=f"ledger[{index}].trade_id"
            )
            if trade_id not in active:
                raise DojoBotTrainerError("ledger exit has no active trade")
            position = active[trade_id]
            close_units = _number(
                payload.get("units", position["units"]),
                field=f"ledger[{index}].units",
                positive=True,
            )
            if close_units > float(position["units"]) + 1e-9:
                raise DojoBotTrainerError("ledger exit over-closes a trade")
            closed = _sim_quote_time(payload, field=f"ledger[{index}]")
            if closed < start or closed > end or closed < position["opened"]:
                raise DojoBotTrainerError("ledger exit time is invalid")
            held_hours = (closed - position["opened"]).total_seconds() / 3600.0
            capital_lock_jpy_hours += (
                close_units * float(position["margin_per_unit"]) * held_hours
            )
            unit_hold_hours += close_units * held_hours
            closed_units += close_units
            pnl = _number(payload.get("pl_jpy"), field=f"ledger[{index}].pl_jpy")
            pair = str(position["pair"])
            pair_pnl[pair] += pnl
            realized_balance += pnl
            realized_peak = max(realized_peak, realized_balance)
            if realized_peak > 0:
                realized_drawdown = max(
                    realized_drawdown,
                    (realized_peak - realized_balance) / realized_peak,
                )
            position["units"] = float(position["units"]) - close_units
            if position["units"] <= 1e-9:
                del active[trade_id]
                pair_active[pair] -= 1
            resolved_exit_slices += 1
            if event == "MARGIN_CLOSEOUT":
                margin_closeouts += 1
        elif event == "ACCOUNT_MARK":
            account = payload.get("account")
            if not isinstance(account, Mapping):
                raise DojoBotTrainerError("ACCOUNT_MARK account is missing")
            equity = _number(account.get("equity_jpy"), field="ACCOUNT_MARK.equity")
            margin_used = _number(
                account.get("margin_used_jpy"),
                field="ACCOUNT_MARK.margin_used",
                non_negative=True,
            )
            mtm_equity_peak = max(mtm_equity_peak, equity)
            if mtm_equity_peak > 0:
                mtm_drawdown = max(
                    mtm_drawdown,
                    (mtm_equity_peak - equity) / mtm_equity_peak,
                )
            if equity > 0:
                peak_margin_fraction = max(peak_margin_fraction, margin_used / equity)
            account_marks += 1

        margin_now = sum(
            float(item["units"]) * float(item["margin_per_unit"])
            for item in active.values()
        )
        peak_margin_jpy = max(peak_margin_jpy, margin_now)
        if realized_balance > 0:
            peak_margin_fraction = max(
                peak_margin_fraction, margin_now / realized_balance
            )

    terminal_payload = last_record["payload"]
    terminal_account = terminal_payload.get("account")
    if not isinstance(terminal_account, Mapping):
        raise DojoBotTrainerError("SESSION_STOP account is missing")
    terminal_balance = _number(
        terminal_account.get("balance_jpy"), field="SESSION_STOP.balance"
    )
    open_positions = _integer(
        terminal_account.get("open_positions"),
        field="SESSION_STOP.open_positions",
        non_negative=True,
    )
    resting_orders = _integer(
        terminal_account.get("resting_orders"),
        field="SESSION_STOP.resting_orders",
        non_negative=True,
    )
    terminal_flat = not active and open_positions == 0 and resting_orders == 0
    if terminal_flat and not math.isclose(
        terminal_balance,
        realized_balance,
        rel_tol=0,
        abs_tol=max(0.05, resolved_exit_slices * 0.01),
    ):
        raise DojoBotTrainerError("ledger realized balance does not reconcile terminal")
    terminal_net = terminal_balance - start_balance
    if not math.isclose(
        sum(pair_pnl.values()),
        terminal_net,
        rel_tol=0,
        abs_tol=max(0.05, resolved_exit_slices * 0.01),
    ):
        raise DojoBotTrainerError("ledger pair PnL does not reconcile terminal net")

    positive_total = sum(max(0.0, value) for value in pair_pnl.values())
    positive_shares = {
        pair: max(0.0, pnl) / positive_total if positive_total > 0 else 0.0
        for pair, pnl in pair_pnl.items()
    }
    positive_hhi = (
        sum(share * share for share in positive_shares.values())
        if positive_total > 0
        else None
    )
    additive_lopo = {pair: terminal_net - pnl for pair, pnl in pair_pnl.items()}
    attempts = fill_count + margin_rejections
    unverified_mtm_claim_present = bool(
        account_marks
        and terminal_payload.get("mtm_complete") is True
        and terminal_payload.get("mtm_mark_count") == account_marks
    )
    # Legacy rows remain diagnostic-only.  A declared V1 contract reaches this
    # point only after independent coordinate/state/account reconstruction.
    mtm_complete = verified_mtm is not None
    if verified_mtm is not None:
        account_marks = int(verified_mtm["mtm_mark_count"])
        mtm_drawdown = float(verified_mtm["mtm_max_drawdown_fraction"])
        peak_margin_jpy = float(verified_mtm["peak_margin_jpy"])
        peak_margin_fraction = float(verified_mtm["peak_margin_fraction"])
    window_hours = (end - start).total_seconds() / 3600.0
    body = {
        "contract": LEDGER_METRICS_CONTRACT,
        "schema_version": 1,
        "ledger_path": ledger_label,
        "ledger_size_bytes": artifact_stats["ledger_size"],
        "ledger_file_sha256": artifact_stats["ledger_file_sha256"],
        "ledger_terminal_sha256": artifact_stats["ledger_terminal_sha256"],
        "ledger_record_count": artifact_stats["ledger_record_count"],
        "reproducibility_manifest_sha256": manifest_sha,
        "corpus_sha256": corpus_sha,
        "intrabar": expected_intrabar,
        "hardened_costs": {
            "slippage_pips_per_fill": expected_slippage,
            "financing_pips_per_day": expected_financing,
        },
        "strategy_owner_id": expected_owner,
        "bot_module_sha256": expected_module_sha,
        "bot_dependency_sha256": expected_dependencies,
        "bot_config_sha256": expected_config_sha,
        "strategy_execution_trace_status": (
            "SOURCE_BOUND_TRUSTED_RUNNER_NOT_INDEPENDENTLY_REEXECUTED"
        ),
        "expected_pairs": pairs,
        "window_start_utc": start.isoformat(),
        "window_end_utc": end.isoformat(),
        "start_balance_jpy": round(start_balance, 8),
        "terminal_balance_jpy": round(terminal_balance, 8),
        "terminal_net_jpy": round(terminal_net, 8),
        "terminal_flat": terminal_flat,
        "fill_count": fill_count,
        "resolved_exit_slices": resolved_exit_slices,
        "margin_closeouts": margin_closeouts,
        "realized_max_drawdown_fraction": round(realized_drawdown, 12),
        "mtm_complete": mtm_complete,
        "mtm_evidence_status": (
            str(verified_mtm["mtm_evidence_status"])
            if verified_mtm is not None
            else "UNVERIFIED_ACCOUNT_MARK_SEQUENCE"
            if account_marks or unverified_mtm_claim_present
            else "NO_ACCOUNT_MARK_SEQUENCE"
        ),
        "unverified_mtm_claim_present": unverified_mtm_claim_present,
        "mtm_mark_count": account_marks,
        "mtm_max_drawdown_fraction": round(mtm_drawdown, 12) if mtm_complete else None,
        "peak_entry_margin_estimate_jpy": round(peak_margin_jpy, 8),
        "peak_entry_margin_estimate_fraction": round(peak_margin_fraction, 12),
        "margin_reject_count": margin_rejections,
        "margin_reject_rate": round(
            margin_rejections / attempts if attempts else 1.0, 12
        ),
        "pair_pnl_jpy": {pair: round(value, 8) for pair, value in pair_pnl.items()},
        "pair_entries": pair_entries,
        "pair_peak_active": pair_peak_active,
        "positive_pair_share": {
            pair: round(value, 12) for pair, value in positive_shares.items()
        },
        "positive_pair_max_share": round(max(positive_shares.values()), 12),
        "positive_pair_hhi": round(positive_hhi, 12)
        if positive_hhi is not None
        else None,
        "effective_positive_pairs": round(1.0 / positive_hhi, 12)
        if positive_hhi not in {None, 0.0}
        else None,
        "leave_one_pair_out_net_jpy": {
            pair: round(value, 8) for pair, value in additive_lopo.items()
        },
        "lopo_basis": "ADDITIVE_ATTRIBUTION_NOT_COUNTERFACTUAL_REPLAY",
        "lopo_replay_complete": False,
        "capital_lock_margin_jpy_hours": round(capital_lock_jpy_hours, 8),
        "average_margin_fraction_over_window": round(
            capital_lock_jpy_hours / (start_balance * window_hours), 12
        ),
        "unit_weighted_hold_hours": round(unit_hold_hours / closed_units, 12)
        if closed_units
        else None,
        "classification": "WORN_HISTORICAL_TRAIN_LEDGER_DIAGNOSTIC_ONLY",
        "proof_eligible": False,
        "promotion_eligible": False,
        "live_permission": False,
        "order_authority": "NONE",
        "broker_mutation_allowed": False,
    }
    # The display-only locator differs when the runner scores by absolute path
    # and the verifier re-opens the same artifact through its trusted relative
    # root.  Keep the receipt digest relocatable; file bytes and all economics
    # remain bound by the adjacent size/file/hash-chain fields.
    digest_body = {key: value for key, value in body.items() if key != "ledger_path"}
    return {**body, "metrics_sha256": _canonical_sha256(digest_body)}


__all__ = [
    "CELL_CONTRACT",
    "DojoBotTrainerError",
    "EVALUATION_CONTRACT",
    "LEDGER_METRICS_CONTRACT",
    "MAX_CANDIDATES",
    "PROPOSAL_CONTRACT",
    "SEALED_STUDY_CONTRACT",
    "STUDY_CONTRACT",
    "evaluate_training",
    "score_ledger_metrics",
    "seal_candidate_proposal",
    "seal_cell_result",
    "seal_study",
    "verify_sealed_study",
]
