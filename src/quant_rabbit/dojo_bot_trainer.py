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

import hashlib
import json
import math
import os
import stat
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
_FILL_EVENTS = {"FILL_MARKET", "FILL_LIMIT", "FILL_STOP"}
_EXIT_EVENTS = {"EXIT_TP", "EXIT_SL", "CLOSE", "MARGIN_CLOSEOUT"}
_MARGIN_REJECTION_EVENTS = {
    "ORDER_REJECTED_INSUFFICIENT_MARGIN",
    "LIMIT_REJECTED_INSUFFICIENT_MARGIN",
}
_OWNED_LEDGER_EVENTS = (
    _FILL_EVENTS | _EXIT_EVENTS | _MARGIN_REJECTION_EVENTS | {"PERIOD_END_SETTLEMENT"}
)


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

    supplied_handle = hasattr(ledger_path, "read")
    path: Path | None = None
    local_handle: BinaryIO | None = None
    if supplied_handle:
        handle = ledger_path
        if not hasattr(handle, "readline"):
            raise DojoBotTrainerError("ledger stream is not readable")
        ledger_label = ledger_artifact_path or "<verified-ledger-stream>"
    else:
        path = Path(ledger_path)
        try:
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise DojoBotTrainerError("ledger file must be a regular non-symlink")
            flags = (
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(path, flags)
            local_handle = os.fdopen(descriptor, "rb")
        except DojoBotTrainerError:
            raise
        except OSError as exc:
            raise DojoBotTrainerError("ledger file is missing") from exc
        handle = local_handle
        ledger_label = str(path.resolve())

    records: list[dict[str, Any]] = []
    previous = _ZERO_SHA256
    ledger_file_digest = hashlib.sha256()
    ledger_size = 0
    try:
        for line_number, raw_line in enumerate(handle, start=1):
            ledger_size += len(raw_line)
            if ledger_size > _MAX_LEDGER_BYTES:
                raise DojoBotTrainerError("ledger artifact exceeds the size limit")
            ledger_file_digest.update(raw_line)
            if not raw_line.strip():
                raise DojoBotTrainerError(f"blank ledger row at line {line_number}")
            if len(raw_line) > MAX_JSON_BYTES:
                raise DojoBotTrainerError("ledger row exceeds the JSON size limit")
            row = _load_jsonish(raw_line, field=f"ledger[{line_number}]")
            row = dict(
                _require_exact_keys(
                    row,
                    field=f"ledger[{line_number}]",
                    expected={"ts_utc", "event", "payload", "prev_sha", "sha"},
                )
            )
            _parse_utc(row["ts_utc"], field=f"ledger[{line_number}].ts_utc")
            if not isinstance(row["event"], str) or not isinstance(
                row["payload"], Mapping
            ):
                raise DojoBotTrainerError("ledger event/payload is malformed")
            if row["prev_sha"] != previous:
                raise DojoBotTrainerError("ledger hash chain predecessor mismatch")
            claimed = _sha256(row["sha"], field=f"ledger[{line_number}].sha")
            body = {key: item for key, item in row.items() if key != "sha"}
            if claimed != _canonical_sha256(body):
                raise DojoBotTrainerError("ledger row digest mismatch")
            previous = claimed
            records.append(row)
            if (
                row["event"] in _OWNED_LEDGER_EVENTS
                and row["payload"].get("strategy_owner_id") != expected_owner
            ):
                raise DojoBotTrainerError(
                    "owned ledger event has missing or mismatched strategy owner"
                )
    finally:
        if local_handle is not None:
            local_handle.close()
    if not records or records[0]["event"] != "SESSION_START":
        raise DojoBotTrainerError("ledger must begin with SESSION_START")
    if records[-1]["event"] != "SESSION_STOP":
        raise DojoBotTrainerError("ledger must end with SESSION_STOP")
    if (
        sum(row["event"] == "SESSION_START" for row in records) != 1
        or sum(row["event"] == "SESSION_STOP" for row in records) != 1
    ):
        raise DojoBotTrainerError("ledger session boundaries are not unique")

    start_payload = records[0]["payload"]
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
    loaded = [row for row in records if row["event"] == "BOT_LOADED"]
    if len(loaded) != 1:
        raise DojoBotTrainerError("ledger requires exactly one BOT_LOADED receipt")
    loaded_payload = loaded[0]["payload"]
    if (
        loaded_payload.get("strategy_owner_id") != expected_owner
        or loaded_payload.get("class") != "Bot"
        or loaded_payload.get("module") != bot.get("module_path")
    ):
        raise DojoBotTrainerError("BOT_LOADED receipt does not match manifest bot")
    settlements = [row for row in records if row["event"] == "PERIOD_END_SETTLEMENT"]
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

    for index, row in enumerate(records[1:-1], start=2):
        event = row["event"]
        payload = row["payload"]
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

    terminal_payload = records[-1]["payload"]
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
        abs_tol=max(0.05, fill_count * 0.01),
    ):
        raise DojoBotTrainerError("ledger realized balance does not reconcile terminal")
    terminal_net = terminal_balance - start_balance
    if not math.isclose(
        sum(pair_pnl.values()),
        terminal_net,
        rel_tol=0,
        abs_tol=max(0.05, fill_count * 0.01),
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
    # The current virtual-market runner does not yet emit the contracted,
    # coordinate-complete ACCOUNT_MARK sequence.  A terminal self-claim plus
    # one or more arbitrary marks is not enough to prove continuous MTM, so
    # remain fail-closed until mark index/phase/pair coverage is verified.
    mtm_complete = False
    window_hours = (end - start).total_seconds() / 3600.0
    body = {
        "contract": LEDGER_METRICS_CONTRACT,
        "schema_version": 1,
        "ledger_path": ledger_label,
        "ledger_size_bytes": ledger_size,
        "ledger_file_sha256": ledger_file_digest.hexdigest(),
        "ledger_terminal_sha256": previous,
        "ledger_record_count": len(records),
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
            "UNVERIFIED_ACCOUNT_MARK_SEQUENCE"
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
