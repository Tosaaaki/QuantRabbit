#!/usr/bin/env python3
"""Focused contract tests for the isolated V5 offline replay."""
from __future__ import annotations

import ast
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "qr_v5_session_break", HERE / "replay_session_break_response.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_bar(time: int, o: float, h: float, l: float, c: float, volume: int = 10,
             spread: float = 0.0002) -> MODULE.Bar:
    half = spread / 2.0
    bid = (o - half, h - half, l - half, c - half)
    ask = (o + half, h + half, l + half, c + half)
    return MODULE.Bar(time, bid, ask, volume)


def path_bars(start: int, closes: list[float], highs=None, lows=None) -> list[MODULE.Bar]:
    highs = highs or [value + 0.0002 for value in closes]
    lows = lows or [value - 0.0002 for value in closes]
    rows = []
    prior = closes[0]
    for index, close in enumerate(closes):
        rows.append(make_bar(start + index * 300, prior, highs[index], lows[index], close))
        prior = close
    return rows


class PreregistrationTests(unittest.TestCase):
    def test_exact_family_and_authority(self):
        prereg = MODULE.load_preregistration()
        self.assertTrue(all(MODULE.validate_preregistration(prereg).values()))
        self.assertEqual(len(prereg["family"]["configs"]), 128)
        self.assertEqual(sum(row["selection_eligible"] for row in prereg["family"]["configs"]), 32)
        self.assertEqual(prereg["authority"]["network_attempts_allowed"], 0)
        self.assertEqual(prereg["authority"]["credential_reads_allowed"], 0)
        self.assertEqual(prereg["authority"]["external_orders_allowed"], 0)

    def test_any_frozen_prereg_mutation_fails_canonical_binding(self):
        original = MODULE.load_preregistration()
        mutations = []
        row = copy.deepcopy(original)
        row["selection"]["bootstrap_seed"] += 1
        mutations.append(row)
        row = copy.deepcopy(original)
        row["selection"]["density_floor"]["trades"] -= 1
        mutations.append(row)
        row = copy.deepcopy(original)
        row["execution_arms"]["base_slippage_pips_per_side"] = 0.0
        mutations.append(row)
        row = copy.deepcopy(original)
        row["portfolio_and_reporting"]["gross_leverage_observation_cap"] = 21.0
        mutations.append(row)
        row = copy.deepcopy(original)
        row["family"]["configs"][0]["horizon_bars"] = 999
        mutations.append(row)
        row = copy.deepcopy(original)
        row["family"]["dimensions"]["session"] = ["LONDON_FIX"]
        mutations.append(row)
        for mutated in mutations:
            with self.assertRaisesRegex(ValueError, "canonical SHA-256 mismatch"):
                MODULE.validate_preregistration(mutated)

    def test_semantic_decoder_checks_boundary_before_price(self):
        row = {
            "schema": "QR_OANDA_HISTORICAL_M5_BA_ROW_V1",
            "instrument": "EUR_USD",
            "granularity": "M5",
            "price_component": "BA",
            "complete": True,
            "time_utc": "2025-08-28T04:05:00.000000Z",
            "volume": "not-an-int",
            "bid": "must-not-decode",
            "ask": "must-not-decode",
        }
        with self.assertRaisesRegex(AssertionError, "outside authorized"):
            MODULE.parse_bar(
                json.dumps(row).encode(), "EUR_USD",
                MODULE.parse_time("2025-05-28T04:05:00.000000Z"),
                MODULE.parse_time("2025-08-28T04:05:00.000000Z"),
            )

    def test_exact_prefix_does_not_read_suffix(self):
        prefix = b'{"safe":true}\n'
        suffix = b'{"future_price":"MUST_NOT_BE_READ"}\n'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.jsonl"
            path.write_bytes(prefix + suffix)
            loaded = MODULE.read_exact_prefix(
                path, len(prefix), hashlib.sha256(prefix).hexdigest()
            )
        self.assertEqual(loaded, prefix)
        self.assertNotIn(b"future_price", loaded)

    def test_runner_has_no_network_or_process_management_import(self):
        tree = ast.parse((HERE / "replay_session_break_response.py").read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertTrue(imported.isdisjoint({
            "requests", "urllib", "httpx", "socket", "websocket",
            "subprocess", "keyring",
        }))


class SessionAndGeometryTests(unittest.TestCase):
    def test_london_schedule_is_dst_aware(self):
        _, winter = MODULE.schedule("LONDON_FIX", dt.date(2025, 1, 15))
        _, summer = MODULE.schedule("LONDON_FIX", dt.date(2025, 7, 15))
        winter_hour = dt.datetime.fromtimestamp(winter[0], tz=dt.timezone.utc).hour
        summer_hour = dt.datetime.fromtimestamp(summer[0], tz=dt.timezone.utc).hour
        self.assertEqual(winter_hour, 12)
        self.assertEqual(summer_hour, 11)

    def test_accept_formula_and_structure(self):
        start = 1_700_000_000
        reference = [make_bar(start + i * 300, 1.0, 1.01, 0.99, 1.0) for i in range(48)]
        closes = [1.0 + i * 0.0003 for i in range(46)] + [1.011, 1.012]
        event = path_bars(start + 48 * 300, closes)
        observation = MODULE.make_observation(
            "EUR_USD", "LONDON_FIX", "2025-01-01", reference, event
        )
        self.assertEqual(observation.mode, "ACCEPT_CONTINUATION")
        self.assertEqual(observation.break_side, 1)
        self.assertEqual(observation.trade_side, "LONG")
        self.assertGreater(observation.displacement, 0)
        self.assertGreater(observation.geometry, 0)

    def test_reject_formula_and_structure(self):
        start = 1_700_000_000
        reference = [make_bar(start + i * 300, 1.0, 1.01, 0.99, 1.0) for i in range(48)]
        closes = [1.0] * 44 + [1.009, 1.006, 1.003, 1.001]
        highs = [1.0002] * 20 + [1.012] + [1.009] * 27
        lows = [0.9998] * 48
        event = path_bars(start + 48 * 300, closes, highs, lows)
        observation = MODULE.make_observation(
            "USD_JPY", "LONDON_FIX", "2025-01-01", reference, event
        )
        self.assertEqual(observation.mode, "REJECT_FADE")
        self.assertEqual(observation.break_side, 1)
        self.assertEqual(observation.trade_side, "SHORT")
        self.assertGreater(observation.persist_or_reverse, 0)

    def test_both_rails_touched_is_ambiguous(self):
        start = 1_700_000_000
        reference = [make_bar(start + i * 300, 1.0, 1.01, 0.99, 1.0) for i in range(48)]
        closes = [1.0] * 48
        event = path_bars(
            start + 48 * 300, closes,
            [1.02] + [1.001] * 47,
            [0.98] + [0.999] * 47,
        )
        observation = MODULE.make_observation(
            "AUD_USD", "LONDON_FIX", "2025-01-01", reference, event
        )
        self.assertTrue(observation.ambiguous)
        self.assertIsNone(observation.mode)


class ExecutionAndStatisticsTests(unittest.TestCase):
    def _observation(self, decision: int) -> MODULE.Observation:
        return MODULE.Observation(
            "EUR_USD", "LONDON_FIX", "2025-01-01", decision,
            "ACCEPT_CONTINUATION", 1, "LONG", 1.0, 1.01,
            0.01, 1.2, 0.8, 0.8, 0.8, 0.8, 100,
            activity=1.2, breadth=0.8, common_usd_sign=-1,
            usd_component=-1.0,
        )

    def _config(self, horizon=24):
        prereg = MODULE.load_preregistration()
        return next(
            row for row in prereg["family"]["configs"]
            if row["session"] == "LONDON_FIX"
            and row["mode"] == "ACCEPT_CONTINUATION"
            and row["breadth"] == "ANY" and row["activity"] == "ANY"
            and row["horizon_bars"] == horizon
        )

    def test_entry_is_strictly_later_and_arms_share_lineage(self):
        decision = 1_700_000_000
        times = range(decision, decision + 27 * 300, 300)
        eur = [make_bar(time, 1.1, 1.101, 1.099, 1.1 + index * 0.0001)
               for index, time in enumerate(times)]
        jpy = [make_bar(time, 145.0, 145.1, 144.9, 145.0, spread=0.02)
               for time in times]
        maps = {
            "EUR_USD": {bar.time: bar for bar in eur},
            "USD_JPY": {bar.time: bar for bar in jpy},
            "AUD_USD": {},
        }
        trade, reason = MODULE.make_trade(
            self._observation(decision), self._config(), maps,
            decision + 100 * 300, MODULE.load_preregistration(),
        )
        self.assertIsNone(reason)
        self.assertEqual(trade["entry_time"], decision + 300)
        self.assertEqual(trade["exit_time"], decision + 25 * 300)
        self.assertGreater(trade["raw_pips"], trade["base_pips"])
        self.assertGreater(trade["base_pips"], trade["adverse_pips"])
        self.assertEqual(len(trade["signal_id"]), 64)
        self.assertEqual(len(trade["lineage_id"]), 64)

    def test_exact_time_jpy_conversion_fails_closed(self):
        decision = 1_700_000_000
        times = range(decision, decision + 27 * 300, 300)
        eur = [make_bar(time, 1.1, 1.101, 1.099, 1.1) for time in times]
        maps = {
            "EUR_USD": {bar.time: bar for bar in eur},
            "USD_JPY": {},
            "AUD_USD": {},
        }
        trade, reason = MODULE.make_trade(
            self._observation(decision), self._config(), maps,
            decision + 100 * 300, MODULE.load_preregistration(),
        )
        self.assertIsNone(trade)
        self.assertEqual(reason, "JPY_CONVERSION_GAP")

    def test_common_block_resamples_are_deterministic(self):
        first = MODULE.common_block_weights(31, 100, 7, 5)
        second = MODULE.common_block_weights(31, 100, 7, 5)
        self.assertTrue((first == second).all())
        self.assertTrue((first.sum(axis=1) == 31).all())

    def test_max_t_requires_and_records_all_128_standardized_columns(self):
        prereg = MODULE.load_preregistration()
        configs = prereg["family"]["configs"]
        day_count = 20
        base = MODULE.np.linspace(-1.0, 1.0, day_count)
        sums = MODULE.np.vstack([base + index * 0.001 for index in range(128)])
        counts = MODULE.np.ones_like(sums)
        weights = MODULE.common_block_weights(day_count, 100, 9, 5)
        lcbs, critical, audit = MODULE.max_t_lcbs(configs, sums, counts, weights)
        self.assertEqual(len(lcbs), 128)
        self.assertTrue(MODULE.math.isfinite(critical))
        self.assertEqual(audit["family_count"], 128)
        self.assertEqual(audit["standardized_count"], 128)
        broken_counts = counts.copy()
        broken_counts[127, :] = 0.0
        with self.assertRaisesRegex(ValueError, "MAX_T_ALL_128_STANDARDIZATION_FAILED"):
            MODULE.max_t_lcbs(configs, sums, broken_counts, weights)

    def test_exact_ablation_changes_only_breadth_and_activity(self):
        prereg = MODULE.load_preregistration()
        selected = next(row for row in prereg["family"]["configs"] if row["selection_eligible"])
        ablation = MODULE.exact_ablation(selected, prereg["family"]["configs"])
        for field in ("session", "mode", "displacement_quantile", "geometry_quantile", "horizon_bars"):
            self.assertEqual(selected[field], ablation[field])
        self.assertEqual(ablation["breadth"], "ANY")
        self.assertEqual(ablation["activity"], "ANY")


class SealedFailureEvidenceTests(unittest.TestCase):
    def test_attempt_one_failure_evidence_is_self_consistent(self):
        result = json.loads((HERE / "result.json").read_text())
        packet = json.loads((HERE / "evidence_packet.json").read_text())
        self.assertEqual(result["status"], "REJECTED_DISCOVERY_FAMILY_UNSTANDARDIZABLE")
        self.assertEqual(result["attempt"]["attempt_number"], 1)
        self.assertEqual(result["attempt"]["exit_code"], 1)
        self.assertFalse(result["attempt"]["rerun_permitted_for_this_candidate"])
        family = result["discovery_family_standardization"]
        self.assertEqual(family["family_count_required"], 128)
        self.assertEqual(family["configs_with_at_least_one_observation"], 108)
        self.assertEqual(family["zero_observation_config_count"], 20)
        self.assertEqual(len(set(family["zero_observation_config_ids"])), 20)
        self.assertEqual(result["decode_audit"]["validation_rows_decoded"], 0)
        self.assertEqual(result["decode_audit"]["holdout_rows_decoded"], 0)
        self.assertEqual(result["authority"]["external_orders"], 0)
        claimed_result = result.pop("result_sha256")
        self.assertEqual(claimed_result, hashlib.sha256(MODULE.canonical(result)).hexdigest())
        claimed_packet = packet.pop("packet_sha256")
        self.assertEqual(claimed_packet, hashlib.sha256(MODULE.canonical(packet)).hexdigest())
        self.assertEqual(packet["result_sha256"], claimed_result)
        self.assertEqual(packet["validation_rows_decoded"], 0)
        self.assertFalse(packet["profit_proven"])


if __name__ == "__main__":
    unittest.main()
