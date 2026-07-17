from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from quant_rabbit.fast_bot_profiles import (
    AI_ORDER_AUTHORITY,
    LEGACY_HORIZON_LANE,
    LEGACY_PROFILE_ID,
    ORDER_AUTHORITY,
    PRIMARY_ELIGIBLE,
    SHADOW_ONLY,
    LaneKey,
    ProfileContractError,
    build_fallback_edge,
    build_fast_bot_profile,
    build_fast_bot_profile_catalog,
    build_legacy_shadow_only_catalog,
    canonical_sha256,
    legacy_shadow_only_profile,
    validate_fast_bot_profile_catalog,
)
from quant_rabbit.instruments import DEFAULT_TRADER_PAIRS


def _profile(
    profile_id: str,
    *,
    pairs: tuple[str, ...] = ("EUR_USD", "GBP_USD"),
    horizons: tuple[str, ...] = ("RESEARCH_HORIZON_A",),
    eligibility: str = PRIMARY_ELIGIBLE,
):
    return build_fast_bot_profile(
        profile_id=profile_id,
        evaluator_id=f"{profile_id}_EVALUATOR",
        implementation_ref=f"tests.{profile_id}:evaluate",
        supported_pairs=pairs,
        horizon_lanes=horizons,
        activation_eligibility=eligibility,
    )


class _AliasKey(str):
    pass


class FastBotProfilesTest(unittest.TestCase):
    def test_lane_key_is_lossless_and_rejects_aliases(self) -> None:
        lane = LaneKey("EUR_USD", "RESEARCH_PROFILE_A_V1", "RESEARCH_HORIZON_A")
        self.assertEqual(LaneKey.parse(lane.canonical()), lane)

        aliases = (
            ("eur_usd", "RESEARCH_PROFILE_A_V1", "RESEARCH_HORIZON_A"),
            ("EURUSD", "RESEARCH_PROFILE_A_V1", "RESEARCH_HORIZON_A"),
            (" EUR_USD", "RESEARCH_PROFILE_A_V1", "RESEARCH_HORIZON_A"),
            ("ＥＵＲ_USD", "RESEARCH_PROFILE_A_V1", "RESEARCH_HORIZON_A"),
            ("EUR_USD", "research_profile_a_v1", "RESEARCH_HORIZON_A"),
            ("EUR_USD", "RESEARCH-PROFILE-A", "RESEARCH_HORIZON_A"),
            ("EUR_USD", "RESEARCH_PROFILE_A_V1", "research_horizon_a"),
        )
        for pair, profile_id, horizon in aliases:
            with self.subTest(pair=pair, profile_id=profile_id, horizon=horizon):
                with self.assertRaises(ProfileContractError):
                    LaneKey(pair, profile_id, horizon)

        for value in (
            "EUR_USD:RESEARCH_PROFILE_A_V1",
            "EUR_USD:RESEARCH_PROFILE_A_V1:RESEARCH_HORIZON_A:EXTRA",
            1,
        ):
            with self.subTest(value=value):
                with self.assertRaises(ProfileContractError):
                    LaneKey.parse(value)

    def test_legacy_profile_is_immutable_content_addressed_and_shadow_only(
        self,
    ) -> None:
        profile = legacy_shadow_only_profile()
        catalog = build_legacy_shadow_only_catalog()
        payload = catalog.to_dict()

        self.assertEqual(profile.profile_id, LEGACY_PROFILE_ID)
        self.assertEqual(profile.activation_eligibility, SHADOW_ONLY)
        self.assertFalse(profile.primary_eligible)
        self.assertEqual(profile.horizon_lanes, (LEGACY_HORIZON_LANE,))
        self.assertEqual(set(profile.supported_pairs), set(DEFAULT_TRADER_PAIRS))
        self.assertEqual(len(catalog.lanes()), len(DEFAULT_TRADER_PAIRS))
        self.assertEqual(payload["ai_order_authority"], AI_ORDER_AUTHORITY)
        self.assertEqual(payload["order_authority"], ORDER_AUTHORITY)
        self.assertIs(payload["live_permission"], False)
        self.assertIs(payload["broker_mutation_allowed"], False)
        self.assertIs(payload["shadow_only"], True)
        self.assertEqual(validate_fast_bot_profile_catalog(payload), catalog)

        with self.assertRaises(FrozenInstanceError):
            profile.profile_id = "MUTATED"  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            catalog.catalog_sha256 = "a" * 64  # type: ignore[misc]

    def test_catalog_build_is_deterministic_and_detaches_source_mappings(self) -> None:
        profile_a = _profile("RESEARCH_PROFILE_A_V1")
        profile_b = _profile("RESEARCH_PROFILE_B_V1")
        forward = build_fast_bot_profile_catalog((profile_a, profile_b))
        reverse = build_fast_bot_profile_catalog((profile_b, profile_a))
        self.assertEqual(forward.catalog_sha256, reverse.catalog_sha256)
        self.assertEqual(forward.to_dict(), reverse.to_dict())

        source = profile_a.to_dict()
        detached = build_fast_bot_profile_catalog((source, profile_b))
        source["profile_id"] = "MUTATED"
        source["supported_pairs"].append("USD_JPY")
        self.assertEqual(detached, forward)

        reordered_profile = build_fast_bot_profile(
            profile_id="RESEARCH_PROFILE_A_V1",
            evaluator_id="RESEARCH_PROFILE_A_V1_EVALUATOR",
            implementation_ref="tests.RESEARCH_PROFILE_A_V1:evaluate",
            supported_pairs=("GBP_USD", "EUR_USD"),
            horizon_lanes=("RESEARCH_HORIZON_A",),
            activation_eligibility=PRIMARY_ELIGIBLE,
        )
        self.assertEqual(reordered_profile.profile_sha256, profile_a.profile_sha256)

    def test_catalog_tamper_unknown_fields_alias_keys_and_authority_fail_closed(
        self,
    ) -> None:
        catalog = build_fast_bot_profile_catalog((_profile("RESEARCH_PROFILE_A_V1"),))

        tampered = catalog.to_dict()
        tampered["profiles"][0]["evaluator_id"] = "MUTATED_EVALUATOR"
        with self.assertRaises(ProfileContractError):
            validate_fast_bot_profile_catalog(tampered)

        unknown = catalog.to_dict()
        unknown["profiles"][0]["unexpected"] = True
        with self.assertRaises(ProfileContractError):
            validate_fast_bot_profile_catalog(unknown)

        aliased = catalog.to_dict()
        aliased_profile = aliased["profiles"][0]
        aliased_profile[_AliasKey("contract")] = aliased_profile.pop("contract")
        with self.assertRaises(ProfileContractError):
            validate_fast_bot_profile_catalog(aliased)

        for key, unsafe in (
            ("ai_order_authority", "PAIR_SELECTOR"),
            ("order_authority", "ORDER"),
            ("live_permission", True),
            ("broker_mutation_allowed", True),
            ("shadow_only", False),
        ):
            mutated = catalog.to_dict()
            mutated[key] = unsafe
            body = {k: v for k, v in mutated.items() if k != "catalog_sha256"}
            mutated["catalog_sha256"] = canonical_sha256(body)
            with self.subTest(key=key):
                with self.assertRaises(ProfileContractError):
                    validate_fast_bot_profile_catalog(mutated)

    def test_duplicate_and_noncanonical_catalog_members_fail_closed(self) -> None:
        profile_a = _profile("RESEARCH_PROFILE_A_V1")
        profile_b = _profile("RESEARCH_PROFILE_B_V1")
        with self.assertRaisesRegex(ProfileContractError, "duplicate profile_id"):
            build_fast_bot_profile_catalog((profile_a, profile_a))

        external = build_fast_bot_profile_catalog((profile_a, profile_b)).to_dict()
        external["profiles"].reverse()
        body = {k: v for k, v in external.items() if k != "catalog_sha256"}
        external["catalog_sha256"] = canonical_sha256(body)
        with self.assertRaisesRegex(ProfileContractError, "canonical order"):
            validate_fast_bot_profile_catalog(external)

        with self.assertRaisesRegex(ProfileContractError, "duplicates"):
            build_fast_bot_profile(
                profile_id="RESEARCH_PROFILE_C_V1",
                evaluator_id="RESEARCH_PROFILE_C_V1_EVALUATOR",
                implementation_ref="tests.profile_c:evaluate",
                supported_pairs=("EUR_USD", "EUR_USD"),
                horizon_lanes=("RESEARCH_HORIZON_A",),
                activation_eligibility=PRIMARY_ELIGIBLE,
            )

        with self.assertRaises(ProfileContractError):
            build_fast_bot_profile(
                profile_id="RESEARCH_PROFILE_C_V1",
                evaluator_id="RESEARCH_PROFILE_C_V1_EVALUATOR",
                implementation_ref="tests.profile_c:evaluate",
                supported_pairs=("eur_usd",),
                horizon_lanes=("RESEARCH_HORIZON_A",),
                activation_eligibility=PRIMARY_ELIGIBLE,
            )

    def test_unknown_duplicate_and_cyclic_fallbacks_fail_closed(self) -> None:
        profiles = tuple(_profile(f"RESEARCH_PROFILE_{letter}_V1") for letter in "ABC")
        horizon = "RESEARCH_HORIZON_A"

        with self.assertRaisesRegex(ProfileContractError, "unknown profile"):
            build_fast_bot_profile_catalog(
                profiles,
                (
                    build_fallback_edge(
                        pair="EUR_USD",
                        horizon_lane=horizon,
                        from_profile_id="RESEARCH_PROFILE_A_V1",
                        to_profile_id="UNKNOWN_PROFILE_V1",
                    ),
                ),
            )

        duplicate_source = (
            build_fallback_edge(
                pair="EUR_USD",
                horizon_lane=horizon,
                from_profile_id="RESEARCH_PROFILE_A_V1",
                to_profile_id="RESEARCH_PROFILE_B_V1",
            ),
            build_fallback_edge(
                pair="EUR_USD",
                horizon_lane=horizon,
                from_profile_id="RESEARCH_PROFILE_A_V1",
                to_profile_id="RESEARCH_PROFILE_C_V1",
            ),
        )
        with self.assertRaisesRegex(ProfileContractError, "duplicate fallback source"):
            build_fast_bot_profile_catalog(profiles, duplicate_source)

        for edges in (
            (
                build_fallback_edge(
                    pair="EUR_USD",
                    horizon_lane=horizon,
                    from_profile_id="RESEARCH_PROFILE_A_V1",
                    to_profile_id="RESEARCH_PROFILE_A_V1",
                ),
            ),
            (
                build_fallback_edge(
                    pair="EUR_USD",
                    horizon_lane=horizon,
                    from_profile_id="RESEARCH_PROFILE_A_V1",
                    to_profile_id="RESEARCH_PROFILE_B_V1",
                ),
                build_fallback_edge(
                    pair="EUR_USD",
                    horizon_lane=horizon,
                    from_profile_id="RESEARCH_PROFILE_B_V1",
                    to_profile_id="RESEARCH_PROFILE_A_V1",
                ),
            ),
        ):
            with self.subTest(edges=edges):
                with self.assertRaisesRegex(ProfileContractError, "fallback cycle"):
                    build_fast_bot_profile_catalog(profiles, edges)

        legacy = legacy_shadow_only_profile()
        with self.assertRaisesRegex(ProfileContractError, "not primary eligible"):
            build_fast_bot_profile_catalog(
                (profiles[0], legacy),
                (
                    build_fallback_edge(
                        pair="EUR_USD",
                        horizon_lane=LEGACY_HORIZON_LANE,
                        from_profile_id="RESEARCH_PROFILE_A_V1",
                        to_profile_id=LEGACY_PROFILE_ID,
                    ),
                ),
            )

        target_without_lane = _profile(
            "RESEARCH_PROFILE_D_V1",
            pairs=("GBP_USD",),
        )
        with self.assertRaisesRegex(
            ProfileContractError, "fallback target does not support"
        ):
            build_fast_bot_profile_catalog(
                (profiles[0], target_without_lane),
                (
                    build_fallback_edge(
                        pair="EUR_USD",
                        horizon_lane=horizon,
                        from_profile_id="RESEARCH_PROFILE_A_V1",
                        to_profile_id="RESEARCH_PROFILE_D_V1",
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
