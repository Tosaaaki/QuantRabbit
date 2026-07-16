from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rabbit.ai_regime_supervision import (
    CANDIDATE_CONTRACT,
    build_ai_regime_supervision,
    write_ai_regime_supervision,
)
from quant_rabbit.fast_bot import AI_SUPERVISION_CONTRACT, REGIME_CONTRACT


NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)


def _seal(body: dict) -> dict:
    raw = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {**body, "contract_sha256": hashlib.sha256(raw).hexdigest()}


def _artifacts() -> tuple[dict, dict]:
    regime = _seal(
        {
            "contract": REGIME_CONTRACT,
            "schema_version": 1,
            "rows": [{"pair": "EUR_USD"}, {"pair": "USD_JPY"}],
        }
    )
    scorecard = _seal(
        {
            "contract": "QR_FAST_BOT_FORWARD_SCORECARD_V1",
            "schema_version": 1,
            "resolved_signals": 10,
        }
    )
    return regime, scorecard


def _candidate(regime: dict, scorecard: dict) -> dict:
    return {
        "contract": CANDIDATE_CONTRACT,
        "schema_version": 1,
        "reviewed_at_utc": NOW.isoformat(),
        "review_reason": "periodic six-hour regime review",
        "regime_contract_sha256": regime["contract_sha256"],
        "scorecard_contract_sha256": scorecard["contract_sha256"],
        "pairs": {
            "EUR_USD": {
                "mode": "CAUTION",
                "reason": "material volatility transition",
                "expires_at_utc": (NOW + timedelta(hours=3)).isoformat(),
            }
        },
    }


class AiRegimeSupervisionTest(unittest.TestCase):
    def test_sealed_supervision_has_no_order_authority(self) -> None:
        regime, scorecard = _artifacts()
        result = build_ai_regime_supervision(
            _candidate(regime, scorecard),
            regime_contract=regime,
            scorecard=scorecard,
            now_utc=NOW,
        )

        self.assertEqual(result["contract"], AI_SUPERVISION_CONTRACT)
        self.assertEqual(result["ai_order_authority"], "NONE")
        self.assertFalse(result["live_permission"])
        self.assertFalse(result["broker_mutation_allowed"])
        self.assertEqual(result["pairs"]["EUR_USD"]["mode"], "CAUTION")
        self.assertEqual(len(result["contract_sha256"]), 64)

    def test_order_fields_and_stale_bindings_fail_closed(self) -> None:
        regime, scorecard = _artifacts()
        for mutation in (
            {"action": "TRADE"},
            {"units": 1000},
            {"live_permission": True},
        ):
            candidate = {**_candidate(regime, scorecard), **mutation}
            with self.subTest(mutation=mutation):
                with self.assertRaisesRegex(ValueError, "order-authority"):
                    build_ai_regime_supervision(
                        candidate,
                        regime_contract=regime,
                        scorecard=scorecard,
                        now_utc=NOW,
                    )
        stale = {
            **_candidate(regime, scorecard),
            "scorecard_contract_sha256": "a" * 64,
        }
        with self.assertRaisesRegex(ValueError, "scorecard binding"):
            build_ai_regime_supervision(
                stale,
                regime_contract=regime,
                scorecard=scorecard,
                now_utc=NOW,
            )

        invalid_schema = {**_candidate(regime, scorecard), "schema_version": True}
        with self.assertRaisesRegex(ValueError, "contract is invalid"):
            build_ai_regime_supervision(
                invalid_schema,
                regime_contract=regime,
                scorecard=scorecard,
                now_utc=NOW,
            )

    def test_pair_shape_expiry_and_universe_are_bounded(self) -> None:
        regime, scorecard = _artifacts()
        candidate = _candidate(regime, scorecard)
        candidate["pairs"]["EUR_USD"]["expires_at_utc"] = (
            NOW + timedelta(hours=7)
        ).isoformat()
        with self.assertRaisesRegex(ValueError, "within six hours"):
            build_ai_regime_supervision(
                candidate,
                regime_contract=regime,
                scorecard=scorecard,
                now_utc=NOW,
            )

        candidate = _candidate(regime, scorecard)
        candidate["pairs"] = {
            "XAU_USD": candidate["pairs"]["EUR_USD"],
        }
        with self.assertRaisesRegex(ValueError, "unsupported or absent"):
            build_ai_regime_supervision(
                candidate,
                regime_contract=regime,
                scorecard=scorecard,
                now_utc=NOW,
            )

    def test_writer_atomically_replaces_only_the_supervision_artifact(self) -> None:
        regime, scorecard = _artifacts()
        candidate = _candidate(regime, scorecard)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidate_path = root / "candidate.json"
            regime_path = root / "regime.json"
            scorecard_path = root / "scorecard.json"
            output_path = root / "ai_regime_supervision.json"
            for path, value in (
                (candidate_path, candidate),
                (regime_path, regime),
                (scorecard_path, scorecard),
            ):
                path.write_text(json.dumps(value), encoding="utf-8")

            result = write_ai_regime_supervision(
                candidate_path,
                regime_path,
                scorecard_path,
                output_path,
                now_utc=NOW,
            )
            written = json.loads(output_path.read_text())

        self.assertEqual(written, result)
        self.assertEqual(written["ai_order_authority"], "NONE")


if __name__ == "__main__":
    unittest.main()
