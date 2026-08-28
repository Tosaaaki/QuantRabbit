from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_progressive_live_preflight",
    ROOT / "tools" / "run_progressive_live_preflight.py",
)
assert SPEC and SPEC.loader
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


class ProgressiveLivePreflightTests(unittest.TestCase):
    def test_signal_model_uses_broker_minimum_margin_stop_and_factor(self) -> None:
        quotes = {
            "EUR_USD": SimpleNamespace(mid=1.16),
            "USD_JPY": SimpleNamespace(mid=160.0),
        }
        sizing = preflight.signal_sizing_input(
            {
                "pair": "EUR_USD",
                "side": "LONG",
                "entry": 1.16000,
                "stop_loss": 1.15900,
            },
            quotes=quotes,
            instruments={
                "EUR_USD": {
                    "marginRate": "0.04",
                    "minimumTradeSize": "1",
                    "maximumOrderUnits": "100000000",
                    "pipLocation": -4,
                }
            },
            stress_pips=25.0,
        )
        self.assertEqual(sizing.broker_minimum_units, 1)
        self.assertEqual(sizing.requested_units, 100_000_000)
        self.assertAlmostEqual(sizing.margin_jpy_per_unit, 7.424)
        self.assertAlmostEqual(sizing.loss_jpy_per_unit, 0.16)
        self.assertAlmostEqual(sizing.factor_delta_jpy_per_unit["EUR"], 185.6)
        self.assertAlmostEqual(sizing.factor_delta_jpy_per_unit["USD"], -185.6)
        self.assertGreater(
            sizing.stress_closeout_margin_jpy_per_unit,
            sizing.closeout_margin_jpy_per_unit,
        )

    def test_release_requires_exact_seals_and_stays_pre_gateway(self) -> None:
        admission_body = {
            "contract": preflight.FORWARD_ADMISSION_CONTRACT,
            "admission_mode": "PROGRESSIVE_MICRO_LIVE",
            "fixed_sample_wait_required_for_micro_live": False,
            "micro_live_only": True,
        }
        admission = {
            **admission_body,
            "admission_sha256": preflight.canonical_sha(admission_body),
        }
        risk_body = {
            "contract": preflight.RISK_CONTRACT,
            "accepted_by_user": True,
            "acceptance_source": "EXPLICIT_USER_DECISION",
        }
        risk = {**risk_body, "risk_contract_sha256": preflight.canonical_sha(risk_body)}
        manifest = {"software_version_sha256": "a" * 64}
        body = {
            "contract": "QR_PROGRESSIVE_LIVE_RELEASE_RECEIPT_V1",
            "status": "SEALED_AWAITING_FRESH_ACCOUNT_GATE",
            "approval_packet_sha256": "b" * 64,
            "software_manifest": manifest,
            "forward_admission": admission,
            "risk_contract": risk,
            "live_permission": False,
            "broker_mutation_allowed": False,
        }
        release = {**body, "release_receipt_sha256": preflight.canonical_sha(body)}
        admitted, accepted = preflight.verify_release_receipt(
            release,
            approval_packet_sha256="b" * 64,
            software_manifest=manifest,
        )
        self.assertEqual(admitted["admission_mode"], "PROGRESSIVE_MICRO_LIVE")
        self.assertTrue(accepted["accepted_by_user"])
        tampered = json.loads(json.dumps(release))
        tampered["risk_contract"]["accepted_by_user"] = False
        with self.assertRaisesRegex(preflight.PreflightBlocked, "INVALID_OR_DRIFTED"):
            preflight.verify_release_receipt(
                tampered,
                approval_packet_sha256="b" * 64,
                software_manifest=manifest,
            )

    def test_mode_ledger_is_deduplicated_and_corruption_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mode.jsonl"
            event = {"event_id": "mode-1", "mode": "SHADOW_ONLY"}
            self.assertTrue(preflight.append_event_once(path, event))
            self.assertFalse(preflight.append_event_once(path, event))
            self.assertEqual(len(path.read_text().splitlines()), 1)
            path.write_text("not-json\n")
            with self.assertRaisesRegex(preflight.PreflightBlocked, "CORRUPT"):
                preflight.append_event_once(path, {"event_id": "mode-2"})

    def test_unsealed_release_never_grants_live_permission(self) -> None:
        with self.assertRaisesRegex(preflight.PreflightBlocked, "INVALID_OR_DRIFTED"):
            preflight.verify_release_receipt(
                {
                    "contract": "QR_PROGRESSIVE_LIVE_RELEASE_RECEIPT_V1",
                    "status": "SEALED_AWAITING_FRESH_ACCOUNT_GATE",
                    "approval_packet_sha256": "b" * 64,
                    "software_manifest": {"software_version_sha256": "a" * 64},
                    "forward_admission": {},
                    "risk_contract": {},
                    "live_permission": True,
                    "broker_mutation_allowed": True,
                    "release_receipt_sha256": "c" * 64,
                },
                approval_packet_sha256="b" * 64,
                software_manifest={"software_version_sha256": "a" * 64},
            )


if __name__ == "__main__":
    unittest.main()
