from __future__ import annotations

import copy
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from quant_rabbit.fast_bot_profile_selector import (
    FALLBACK,
    NO_PRIMARY,
    PRIMARY_RESEARCH,
    PRIMARY_SELECTED,
    QUARANTINED,
    SHADOW,
    select_fast_bot_profile_lanes,
    validate_fast_bot_profile_routing,
)
from quant_rabbit.fast_bot_profile_state import (
    build_activation_receipt,
    build_quarantine_record,
)
from quant_rabbit.fast_bot_profiles import (
    PRIMARY_ELIGIBLE,
    SHADOW_ONLY,
    LaneKey,
    ProfileContractError,
    build_fallback_edge,
    build_fast_bot_profile,
    build_fast_bot_profile_catalog,
    build_legacy_shadow_only_catalog,
    canonical_sha256,
    validate_fast_bot_profile_catalog,
)


NOW = datetime(2026, 7, 17, 9, 0, tzinfo=timezone.utc)
HORIZON_A = "RESEARCH_HORIZON_A"
HORIZON_B = "RESEARCH_HORIZON_B"
PROFILE_A = "RESEARCH_PROFILE_A_V1"
PROFILE_B = "RESEARCH_PROFILE_B_V1"
PROFILE_C = "RESEARCH_PROFILE_C_V1"
OBSERVER = "LEGACY_OBSERVER_V1"


class _AliasKey(str):
    pass


def _profile(profile_id: str, eligibility: str = PRIMARY_ELIGIBLE):
    return build_fast_bot_profile(
        profile_id=profile_id,
        evaluator_id=f"{profile_id}_EVALUATOR",
        implementation_ref=f"tests.{profile_id}:evaluate",
        supported_pairs=("EUR_USD", "GBP_USD"),
        horizon_lanes=(HORIZON_A, HORIZON_B),
        activation_eligibility=eligibility,
    )


def _catalog():
    profiles = (
        _profile(PROFILE_A),
        _profile(PROFILE_B),
        _profile(PROFILE_C),
        _profile(OBSERVER, SHADOW_ONLY),
    )
    edges = []
    for pair in ("EUR_USD", "GBP_USD"):
        for horizon in (HORIZON_A, HORIZON_B):
            edges.extend(
                (
                    build_fallback_edge(
                        pair=pair,
                        horizon_lane=horizon,
                        from_profile_id=PROFILE_A,
                        to_profile_id=PROFILE_B,
                    ),
                    build_fallback_edge(
                        pair=pair,
                        horizon_lane=horizon,
                        from_profile_id=PROFILE_B,
                        to_profile_id=PROFILE_C,
                    ),
                )
            )
    return build_fast_bot_profile_catalog(profiles, tuple(edges))


def _activation(catalog, pair: str, horizon: str, profile_id: str = PROFILE_A):
    return build_activation_receipt(
        catalog=catalog,
        lane=LaneKey(pair, profile_id, horizon),
        evidence_sha256s=(canonical_sha256({"pair": pair, "horizon": horizon}),),
        activated_at_utc=NOW,
    )


def _quarantine(catalog, pair: str, profile_id: str, horizon: str):
    return build_quarantine_record(
        catalog=catalog,
        lane=LaneKey(pair, profile_id, horizon),
        reason_code="FORWARD_ECONOMICS_FAILED",
        evidence_sha256s=(
            canonical_sha256(
                {"pair": pair, "profile_id": profile_id, "horizon": horizon}
            ),
        ),
        quarantined_at_utc=NOW,
    )


def _scope(routing, pair: str, horizon: str):
    return next(
        row
        for row in routing["scopes"]
        if row["pair"] == pair and row["horizon_lane"] == horizon
    )


def _lane(routing, pair: str, profile_id: str, horizon: str):
    return next(
        row
        for row in routing["lanes"]
        if row["pair"] == pair
        and row["profile_id"] == profile_id
        and row["horizon_lane"] == horizon
    )


def _reseal_routing(routing: dict[str, object]) -> None:
    body = {key: item for key, item in routing.items() if key != "routing_sha256"}
    routing["routing_sha256"] = canonical_sha256(body)


class FastBotProfileSelectorTest(unittest.TestCase):
    def test_no_activation_is_explicit_no_primary_with_all_lanes_shadow(self) -> None:
        catalog = _catalog()
        routing = select_fast_bot_profile_lanes(catalog=catalog)

        self.assertEqual({row["status"] for row in routing["scopes"]}, {NO_PRIMARY})
        self.assertEqual(
            {row["selection_reason"] for row in routing["scopes"]},
            {"NO_ACTIVATION_RECEIPT"},
        )
        self.assertEqual({row["role"] for row in routing["lanes"]}, {SHADOW})
        self.assertTrue(all(row["shadow_enabled"] for row in routing["lanes"]))
        self.assertTrue(
            all(row["deployment_role"] == SHADOW for row in routing["lanes"])
        )
        self.assertEqual(len(routing["lanes"]), len(catalog.lanes()))
        self.assertEqual(
            validate_fast_bot_profile_routing(routing, catalog=catalog), routing
        )

    def test_primary_fallback_shadow_and_quarantine_are_partitioned(self) -> None:
        catalog = _catalog()
        activation = _activation(catalog, "EUR_USD", HORIZON_A)
        quarantine = _quarantine(catalog, "EUR_USD", PROFILE_A, HORIZON_A)
        routing = select_fast_bot_profile_lanes(
            catalog=catalog,
            activation_receipts=(activation,),
            quarantine_records=(quarantine,),
        )
        scope = _scope(routing, "EUR_USD", HORIZON_A)

        self.assertEqual(scope["status"], PRIMARY_SELECTED)
        self.assertEqual(scope["selected_primary_profile_id"], PROFILE_B)
        self.assertEqual(scope["selection_reason"], "SEALED_FALLBACK")
        self.assertEqual(
            _lane(routing, "EUR_USD", PROFILE_A, HORIZON_A)["role"], QUARANTINED
        )
        self.assertEqual(
            _lane(routing, "EUR_USD", PROFILE_B, HORIZON_A)["role"], PRIMARY_RESEARCH
        )
        self.assertEqual(
            _lane(routing, "EUR_USD", PROFILE_C, HORIZON_A)["role"], FALLBACK
        )
        self.assertEqual(_lane(routing, "EUR_USD", OBSERVER, HORIZON_A)["role"], SHADOW)

        scoped = [
            row
            for row in routing["lanes"]
            if row["pair"] == "EUR_USD" and row["horizon_lane"] == HORIZON_A
        ]
        self.assertEqual(
            {row["role"] for row in scoped},
            {PRIMARY_RESEARCH, FALLBACK, SHADOW, QUARANTINED},
        )
        self.assertEqual(
            sum(row["deployment_role"] == PRIMARY_RESEARCH for row in scoped), 1
        )
        for row in scoped:
            self.assertIs(row["shadow_enabled"], True)
            if row["role"] != PRIMARY_RESEARCH:
                self.assertEqual(row["deployment_role"], SHADOW)
            self.assertEqual(row["ai_order_authority"], "NONE")
            self.assertEqual(row["order_authority"], "NONE")
            self.assertIs(row["live_permission"], False)
            self.assertIs(row["broker_mutation_allowed"], False)

        self.assertEqual(
            validate_fast_bot_profile_routing(
                routing,
                catalog=catalog,
                activation_receipts=(activation,),
                quarantine_records=(quarantine,),
            ),
            routing,
        )

    def test_exact_lane_quarantine_does_not_spill_across_any_axis(self) -> None:
        catalog = _catalog()
        activations = (
            _activation(catalog, "EUR_USD", HORIZON_A),
            _activation(catalog, "GBP_USD", HORIZON_A),
            _activation(catalog, "EUR_USD", HORIZON_B),
        )
        quarantine = _quarantine(catalog, "EUR_USD", PROFILE_A, HORIZON_A)
        routing = select_fast_bot_profile_lanes(
            catalog=catalog,
            activation_receipts=activations,
            quarantine_records=(quarantine,),
        )

        self.assertEqual(
            _scope(routing, "EUR_USD", HORIZON_A)["selected_primary_profile_id"],
            PROFILE_B,
        )
        self.assertEqual(
            _scope(routing, "GBP_USD", HORIZON_A)["selected_primary_profile_id"],
            PROFILE_A,
        )
        self.assertEqual(
            _scope(routing, "EUR_USD", HORIZON_B)["selected_primary_profile_id"],
            PROFILE_A,
        )
        self.assertEqual(
            _lane(routing, "EUR_USD", PROFILE_B, HORIZON_A)["role"],
            PRIMARY_RESEARCH,
        )
        self.assertEqual(
            _lane(routing, "EUR_USD", PROFILE_A, HORIZON_A)["role"],
            QUARANTINED,
        )
        for scope in routing["scopes"]:
            primaries = [
                row
                for row in routing["lanes"]
                if row["pair"] == scope["pair"]
                and row["horizon_lane"] == scope["horizon_lane"]
                and row["role"] == PRIMARY_RESEARCH
            ]
            self.assertLessEqual(len(primaries), 1)

    def test_fallback_exhaustion_is_no_primary_not_implicit_default(self) -> None:
        catalog = _catalog()
        activation = _activation(catalog, "EUR_USD", HORIZON_A)
        quarantines = tuple(
            _quarantine(catalog, "EUR_USD", profile_id, HORIZON_A)
            for profile_id in (PROFILE_A, PROFILE_B, PROFILE_C)
        )
        routing = select_fast_bot_profile_lanes(
            catalog=catalog,
            activation_receipts=(activation,),
            quarantine_records=quarantines,
        )
        scope = _scope(routing, "EUR_USD", HORIZON_A)

        self.assertEqual(scope["status"], NO_PRIMARY)
        self.assertIsNone(scope["selected_primary_profile_id"])
        self.assertEqual(scope["selection_reason"], "ALL_SEALED_CANDIDATES_QUARANTINED")
        self.assertEqual(_lane(routing, "EUR_USD", OBSERVER, HORIZON_A)["role"], SHADOW)
        self.assertFalse(
            any(
                row["role"] == PRIMARY_RESEARCH
                for row in routing["lanes"]
                if row["pair"] == "EUR_USD" and row["horizon_lane"] == HORIZON_A
            )
        )

    def test_resolution_is_deterministic_across_source_order(self) -> None:
        catalog = _catalog()
        activations = (
            _activation(catalog, "EUR_USD", HORIZON_A),
            _activation(catalog, "GBP_USD", HORIZON_B),
        )
        quarantines = (
            _quarantine(catalog, "EUR_USD", PROFILE_A, HORIZON_A),
            _quarantine(catalog, "GBP_USD", PROFILE_B, HORIZON_B),
        )
        forward = select_fast_bot_profile_lanes(
            catalog=catalog,
            activation_receipts=activations,
            quarantine_records=quarantines,
        )
        reverse = select_fast_bot_profile_lanes(
            catalog=catalog,
            activation_receipts=tuple(reversed(activations)),
            quarantine_records=tuple(reversed(quarantines)),
        )
        self.assertEqual(forward, reverse)
        self.assertEqual(forward["routing_sha256"], reverse["routing_sha256"])

    def test_tampered_role_partition_alias_unknown_or_authority_fails_closed(
        self,
    ) -> None:
        catalog = _catalog()
        activation = _activation(catalog, "EUR_USD", HORIZON_A)
        routing = select_fast_bot_profile_lanes(
            catalog=catalog, activation_receipts=(activation,)
        )

        mutations = []

        multiple_primary = copy.deepcopy(routing)
        row = _lane(multiple_primary, "EUR_USD", PROFILE_B, HORIZON_A)
        row["role"] = PRIMARY_RESEARCH
        row["deployment_role"] = PRIMARY_RESEARCH
        mutations.append(multiple_primary)

        missing_shadow = copy.deepcopy(routing)
        missing_shadow["lanes"].remove(
            _lane(missing_shadow, "EUR_USD", OBSERVER, HORIZON_A)
        )
        mutations.append(missing_shadow)

        duplicate_lane = copy.deepcopy(routing)
        duplicate_lane["lanes"].append(copy.deepcopy(duplicate_lane["lanes"][0]))
        mutations.append(duplicate_lane)

        duplicate_scope = copy.deepcopy(routing)
        duplicate_scope["scopes"].append(copy.deepcopy(duplicate_scope["scopes"][0]))
        mutations.append(duplicate_scope)

        unknown_profile = copy.deepcopy(routing)
        unknown_profile["lanes"][0]["profile_id"] = "UNKNOWN_PROFILE_V1"
        mutations.append(unknown_profile)

        pair_alias = copy.deepcopy(routing)
        pair_alias["lanes"][0]["pair"] = "eur_usd"
        mutations.append(pair_alias)

        unsafe_lane = copy.deepcopy(routing)
        unsafe_lane["lanes"][0]["live_permission"] = True
        mutations.append(unsafe_lane)

        unsafe_top = copy.deepcopy(routing)
        unsafe_top["ai_order_authority"] = "PROFILE_SELECTOR"
        mutations.append(unsafe_top)

        nested_alias = copy.deepcopy(routing)
        alias_row = nested_alias["lanes"][0]
        role = alias_row.pop("role")
        alias_row[_AliasKey("role")] = role
        mutations.append(nested_alias)

        for mutated in mutations:
            _reseal_routing(mutated)
            with self.subTest(mutation=mutated):
                with self.assertRaises(ProfileContractError):
                    validate_fast_bot_profile_routing(
                        mutated,
                        catalog=catalog,
                        activation_receipts=(activation,),
                    )

    def test_legacy_catalog_can_only_produce_shadow_and_no_primary(self) -> None:
        catalog = build_legacy_shadow_only_catalog()
        routing = select_fast_bot_profile_lanes(catalog=catalog)
        self.assertEqual(len(routing["scopes"]), 28)
        self.assertEqual(len(routing["lanes"]), 28)
        self.assertTrue(all(row["status"] == NO_PRIMARY for row in routing["scopes"]))
        self.assertTrue(all(row["role"] == SHADOW for row in routing["lanes"]))
        self.assertTrue(
            all(not row["research_primary_eligible"] for row in routing["lanes"])
        )

    def test_selection_validates_catalog_once_and_detaches_source_state(self) -> None:
        catalog = _catalog()
        activation = _activation(catalog, "EUR_USD", HORIZON_A)
        quarantine = _quarantine(catalog, "EUR_USD", PROFILE_A, HORIZON_A)

        with (
            patch(
                "quant_rabbit.fast_bot_profile_selector.validate_fast_bot_profile_catalog",
                wraps=validate_fast_bot_profile_catalog,
            ) as selector_validation,
            patch(
                "quant_rabbit.fast_bot_profile_state.validate_fast_bot_profile_catalog",
                wraps=validate_fast_bot_profile_catalog,
            ) as state_validation,
            patch(
                "quant_rabbit.fast_bot_profiles.validate_fast_bot_profile_catalog",
                wraps=validate_fast_bot_profile_catalog,
            ) as nested_validation,
        ):
            routing = select_fast_bot_profile_lanes(
                catalog=catalog,
                activation_receipts=(activation,),
                quarantine_records=(quarantine,),
            )

        self.assertEqual(selector_validation.call_count, 1)
        self.assertEqual(state_validation.call_count, 0)
        self.assertEqual(nested_validation.call_count, 0)
        expected = copy.deepcopy(routing)

        activation["fallback_chain_profile_ids"].clear()
        activation["evidence_sha256s"].clear()
        quarantine["evidence_sha256s"].clear()
        self.assertEqual(routing, expected)


if __name__ == "__main__":
    unittest.main()
