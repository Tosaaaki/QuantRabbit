from __future__ import annotations

import copy
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from quant_rabbit.fast_bot_profile_state import (
    build_activation_receipt,
    build_quarantine_record,
    index_activation_receipts,
    index_quarantine_records,
    validate_activation_receipt,
    validate_quarantine_record,
)
from quant_rabbit.fast_bot_profiles import (
    LEGACY_PROFILE_ID,
    PRIMARY_ELIGIBLE,
    LaneKey,
    ProfileContractError,
    build_fallback_edge,
    build_fast_bot_profile,
    build_fast_bot_profile_catalog,
    canonical_sha256,
    legacy_shadow_only_profile,
    validate_fast_bot_profile_catalog,
)


NOW = datetime(2026, 7, 17, 9, 0, tzinfo=timezone.utc)
HORIZON = "RESEARCH_HORIZON_A"


def _profile(profile_id: str):
    return build_fast_bot_profile(
        profile_id=profile_id,
        evaluator_id=f"{profile_id}_EVALUATOR",
        implementation_ref=f"tests.{profile_id}:evaluate",
        supported_pairs=("EUR_USD", "GBP_USD"),
        horizon_lanes=(HORIZON, "RESEARCH_HORIZON_B"),
        activation_eligibility=PRIMARY_ELIGIBLE,
    )


def _catalog():
    profiles = tuple(_profile(f"RESEARCH_PROFILE_{letter}_V1") for letter in "ABC")
    edges = (
        build_fallback_edge(
            pair="EUR_USD",
            horizon_lane=HORIZON,
            from_profile_id="RESEARCH_PROFILE_A_V1",
            to_profile_id="RESEARCH_PROFILE_B_V1",
        ),
        build_fallback_edge(
            pair="EUR_USD",
            horizon_lane=HORIZON,
            from_profile_id="RESEARCH_PROFILE_B_V1",
            to_profile_id="RESEARCH_PROFILE_C_V1",
        ),
    )
    return build_fast_bot_profile_catalog(profiles, edges)


def _receipt(
    catalog=None,
    *,
    pair: str = "EUR_USD",
    profile_id: str = "RESEARCH_PROFILE_A_V1",
    horizon: str = HORIZON,
):
    catalog = catalog or _catalog()
    return build_activation_receipt(
        catalog=catalog,
        lane=LaneKey(pair, profile_id, horizon),
        evidence_sha256s=("b" * 64, "a" * 64),
        activated_at_utc=NOW,
    )


def _quarantine(
    catalog=None,
    *,
    pair: str = "EUR_USD",
    profile_id: str = "RESEARCH_PROFILE_A_V1",
    horizon: str = HORIZON,
):
    catalog = catalog or _catalog()
    return build_quarantine_record(
        catalog=catalog,
        lane=LaneKey(pair, profile_id, horizon),
        reason_code="FORWARD_ECONOMICS_FAILED",
        evidence_sha256s=("c" * 64,),
        quarantined_at_utc=NOW,
    )


def _reseal(value: dict[str, object], digest_key: str) -> None:
    body = {key: item for key, item in value.items() if key != digest_key}
    value[digest_key] = canonical_sha256(body)


class FastBotProfileStateTest(unittest.TestCase):
    def test_activation_binds_catalog_lane_and_presealed_fallback_chain(self) -> None:
        catalog = _catalog()
        receipt = _receipt(catalog)
        validated = validate_activation_receipt(receipt, catalog)

        self.assertEqual(
            validated["fallback_chain_profile_ids"],
            [
                "RESEARCH_PROFILE_A_V1",
                "RESEARCH_PROFILE_B_V1",
                "RESEARCH_PROFILE_C_V1",
            ],
        )
        self.assertEqual(validated["evidence_sha256s"], ["a" * 64, "b" * 64])
        self.assertEqual(validated["ai_order_authority"], "NONE")
        self.assertEqual(validated["order_authority"], "NONE")
        self.assertIs(validated["live_permission"], False)
        self.assertIs(validated["broker_mutation_allowed"], False)
        self.assertIs(validated["shadow_only"], True)

    def test_activation_tamper_fallback_order_authority_and_catalog_fail_closed(
        self,
    ) -> None:
        catalog = _catalog()
        receipt = _receipt(catalog)

        mutations = (
            ("profile_id", "RESEARCH_PROFILE_B_V1"),
            ("profile_sha256", "d" * 64),
            ("catalog_sha256", "d" * 64),
            ("ai_order_authority", "PROFILE_SELECTOR"),
            ("order_authority", "ORDER"),
            ("live_permission", True),
            ("broker_mutation_allowed", True),
            ("shadow_only", False),
        )
        for key, value in mutations:
            tampered = dict(receipt)
            tampered[key] = value
            _reseal(tampered, "activation_receipt_sha256")
            with self.subTest(key=key):
                with self.assertRaises(ProfileContractError):
                    validate_activation_receipt(tampered, catalog)

        reordered = dict(receipt)
        reordered["fallback_chain_profile_ids"] = list(
            reversed(receipt["fallback_chain_profile_ids"])
        )
        _reseal(reordered, "activation_receipt_sha256")
        with self.assertRaisesRegex(ProfileContractError, "registry sealed"):
            validate_activation_receipt(reordered, catalog)

        noncanonical_evidence = dict(receipt)
        noncanonical_evidence["evidence_sha256s"] = ["b" * 64, "a" * 64]
        _reseal(noncanonical_evidence, "activation_receipt_sha256")
        with self.assertRaisesRegex(ProfileContractError, "not canonical"):
            validate_activation_receipt(noncanonical_evidence, catalog)

        expanded = build_fast_bot_profile_catalog(
            (*catalog.profiles, _profile("RESEARCH_PROFILE_D_V1")),
            catalog.fallback_edges,
        )
        with self.assertRaisesRegex(ProfileContractError, "catalog mismatch"):
            validate_activation_receipt(receipt, expanded)

    def test_activation_rejects_unknown_shadow_only_and_duplicate_scope(self) -> None:
        catalog = _catalog()
        with self.assertRaises(ProfileContractError):
            build_activation_receipt(
                catalog=catalog,
                lane=LaneKey("EUR_USD", "UNKNOWN_PROFILE_V1", HORIZON),
                evidence_sha256s=("a" * 64,),
                activated_at_utc=NOW,
            )

        legacy_catalog = build_fast_bot_profile_catalog((legacy_shadow_only_profile(),))
        with self.assertRaisesRegex(
            ProfileContractError, "not research-primary eligible"
        ):
            build_activation_receipt(
                catalog=legacy_catalog,
                lane=LaneKey(
                    "EUR_USD",
                    LEGACY_PROFILE_ID,
                    legacy_shadow_only_profile().horizon_lanes[0],
                ),
                evidence_sha256s=("a" * 64,),
                activated_at_utc=NOW,
            )

        receipt = _receipt(catalog)
        with self.assertRaisesRegex(ProfileContractError, "multiple activation"):
            index_activation_receipts(catalog, (receipt, receipt))

    def test_quarantine_is_terminal_content_addressed_and_exact_lane(self) -> None:
        catalog = _catalog()
        record = _quarantine(catalog)
        validated = validate_quarantine_record(record, catalog)
        lane = LaneKey(
            validated["pair"],
            validated["profile_id"],
            validated["horizon_lane"],
        )

        self.assertEqual(
            lane,
            LaneKey("EUR_USD", "RESEARCH_PROFILE_A_V1", HORIZON),
        )
        self.assertIs(validated["terminal_for_profile_revision"], True)
        self.assertIs(validated["live_permission"], False)
        self.assertEqual(index_quarantine_records(catalog, (record,)), {lane: record})

        for key, value, reseal in (
            # Pair/horizon remain valid catalog lanes, so their protection is
            # the content address itself rather than an invented signature.
            ("pair", "GBP_USD", False),
            ("profile_id", "RESEARCH_PROFILE_B_V1", True),
            ("horizon_lane", "RESEARCH_HORIZON_B", False),
            ("profile_sha256", "d" * 64, True),
            ("terminal_for_profile_revision", False, True),
            ("ai_order_authority", "PROFILE_SELECTOR", True),
            ("live_permission", True, True),
        ):
            tampered = dict(record)
            tampered[key] = value
            if reseal:
                _reseal(tampered, "quarantine_record_sha256")
            with self.subTest(key=key):
                with self.assertRaises(ProfileContractError):
                    validate_quarantine_record(tampered, catalog)

        with self.assertRaisesRegex(ProfileContractError, "duplicate quarantine"):
            index_quarantine_records(catalog, (record, record))

    def test_state_timestamps_and_evidence_are_strict_and_bounded(self) -> None:
        catalog = _catalog()
        with self.assertRaisesRegex(ProfileContractError, "duplicates"):
            build_activation_receipt(
                catalog=catalog,
                lane=LaneKey("EUR_USD", "RESEARCH_PROFILE_A_V1", HORIZON),
                evidence_sha256s=("a" * 64, "a" * 64),
                activated_at_utc=NOW,
            )

        receipt = _receipt(catalog)
        receipt["activated_at_utc"] = "2026-07-17T09:00:00+00:00"
        _reseal(receipt, "activation_receipt_sha256")
        with self.assertRaisesRegex(ProfileContractError, "canonical UTC"):
            validate_activation_receipt(receipt, catalog)

        with self.assertRaisesRegex(ProfileContractError, "aware datetime"):
            build_quarantine_record(
                catalog=catalog,
                lane=LaneKey("EUR_USD", "RESEARCH_PROFILE_A_V1", HORIZON),
                reason_code="FORWARD_ECONOMICS_FAILED",
                evidence_sha256s=("a" * 64,),
                quarantined_at_utc=datetime(2026, 7, 17, 9, 0),
            )

    def test_validated_and_indexed_state_detaches_all_source_lists(self) -> None:
        catalog = _catalog()
        receipt = _receipt(catalog)
        quarantine = _quarantine(catalog)

        validated_receipt = validate_activation_receipt(receipt, catalog)
        validated_quarantine = validate_quarantine_record(quarantine, catalog)
        indexed_receipt = index_activation_receipts(catalog, (receipt,))[
            ("EUR_USD", HORIZON)
        ]
        indexed_quarantine = index_quarantine_records(catalog, (quarantine,))[
            LaneKey("EUR_USD", "RESEARCH_PROFILE_A_V1", HORIZON)
        ]
        expected = copy.deepcopy(
            (
                validated_receipt,
                validated_quarantine,
                indexed_receipt,
                indexed_quarantine,
            )
        )

        self.assertIsNot(
            receipt["fallback_chain_profile_ids"],
            validated_receipt["fallback_chain_profile_ids"],
        )
        self.assertIsNot(
            receipt["evidence_sha256s"], validated_receipt["evidence_sha256s"]
        )
        self.assertIsNot(
            quarantine["evidence_sha256s"],
            validated_quarantine["evidence_sha256s"],
        )

        receipt["fallback_chain_profile_ids"].clear()
        receipt["evidence_sha256s"].append("f" * 64)
        quarantine["evidence_sha256s"].clear()

        self.assertEqual(
            (
                validated_receipt,
                validated_quarantine,
                indexed_receipt,
                indexed_quarantine,
            ),
            expected,
        )
        for artifact, digest_key in (
            (validated_receipt, "activation_receipt_sha256"),
            (validated_quarantine, "quarantine_record_sha256"),
            (indexed_receipt, "activation_receipt_sha256"),
            (indexed_quarantine, "quarantine_record_sha256"),
        ):
            body = {key: value for key, value in artifact.items() if key != digest_key}
            self.assertEqual(canonical_sha256(body), artifact[digest_key])

    def test_batch_indexes_validate_the_catalog_once_not_once_per_row(self) -> None:
        catalog = _catalog()
        scopes = (
            ("EUR_USD", HORIZON),
            ("EUR_USD", "RESEARCH_HORIZON_B"),
            ("GBP_USD", HORIZON),
            ("GBP_USD", "RESEARCH_HORIZON_B"),
        )
        receipts = tuple(
            _receipt(catalog, pair=pair, horizon=horizon) for pair, horizon in scopes
        )
        quarantines = tuple(
            _quarantine(catalog, pair=pair, horizon=horizon) for pair, horizon in scopes
        )

        target = "quant_rabbit.fast_bot_profile_state.validate_fast_bot_profile_catalog"
        with patch(
            target,
            wraps=validate_fast_bot_profile_catalog,
        ) as validate_catalog:
            self.assertEqual(len(index_activation_receipts(catalog, receipts)), 4)
            self.assertEqual(validate_catalog.call_count, 1)

        with patch(
            target,
            wraps=validate_fast_bot_profile_catalog,
        ) as validate_catalog:
            self.assertEqual(len(index_quarantine_records(catalog, quarantines)), 4)
            self.assertEqual(validate_catalog.call_count, 1)


if __name__ == "__main__":
    unittest.main()
