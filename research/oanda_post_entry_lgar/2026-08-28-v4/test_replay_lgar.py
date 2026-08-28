#!/usr/bin/env python3
"""Focused contract tests for the isolated V4 offline replay."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("qr_v4_replay_lgar", HERE / "replay_lgar.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def bar(when, mid=1.1000, spread=0.0002):
    half = spread / 2.0
    bid = (mid - half, mid + 0.0003 - half, mid - 0.0003 - half, mid - half)
    ask = (mid + half, mid + 0.0003 + half, mid - 0.0003 + half, mid + half)
    return MODULE.Bar(when, bid, ask)


def feature(index, decision_time, side="LONG"):
    return MODULE.Feature(
        index=index,
        decision_time=decision_time,
        trend_side=side,
        path_efficiency=0.8,
        impulse_side=side,
        rail_side=side,
        rail_kind="ACCEPTANCE",
        spread_pips=2.0,
        slot="00:05",
        usd_one_bar_sign=1,
        usd_breadth=1.0,
        usd_breadth_count=3,
    )


def signal(pair, side, decision_time, feature_index=0):
    return MODULE.Signal(
        signal_id=hashlib.sha256(
            f"{pair}|{side}|{decision_time}".encode("utf-8")
        ).hexdigest(),
        feature_hash="f" * 64,
        pair=pair,
        side=side,
        decision_time=decision_time,
        feature_index=feature_index,
    )


def calibration():
    slot_rows = {"00:05": {"rows": 100, "value": 3.0}}
    return {
        "path_efficiency_q67": {
            pair: {"rows": 1000, "value": 0.6}
            for pair in ("EUR_USD", "AUD_USD", "USD_JPY")
        },
        "spread_slot_q75": {
            pair: copy.deepcopy(slot_rows)
            for pair in ("EUR_USD", "AUD_USD", "USD_JPY")
        },
        "mfe_q40_pips": {
            pair: {
                side: {"rows": 1000, "value": 4.0}
                for side in ("LONG", "SHORT")
            }
            for pair in ("EUR_USD", "AUD_USD", "USD_JPY")
        },
    }


def small_market(gapped_eurusd=False):
    start = 1_700_000_000
    exact = [start, start + 300, start + 600, start + 900]
    eur_times = [start, start + 600, start + 900] if gapped_eurusd else exact
    bars_by_pair = {
        "EUR_USD": [bar(when, 1.10 + offset * 0.0001) for offset, when in enumerate(eur_times)],
        "AUD_USD": [bar(when, 0.66 + offset * 0.0001) for offset, when in enumerate(exact)],
        "USD_JPY": [bar(when, 145.0 + offset * 0.01, 0.02) for offset, when in enumerate(exact)],
    }
    features_by_pair = {
        pair: [None] * len(rows) for pair, rows in bars_by_pair.items()
    }
    return start, MODULE.Market(bars_by_pair, features_by_pair)


def simulator(market, signals_by_time=None, policy="P2"):
    prereg = MODULE.load_preregistration()
    return MODULE.PolicySimulator(
        policy,
        market,
        signals_by_time or {},
        calibration(),
        prereg,
        "tuning",
    )


class PreregistrationTests(unittest.TestCase):
    def test_exact_frozen_family_and_zero_authority(self):
        prereg = MODULE.load_preregistration()
        checks = MODULE.validate_preregistration(prereg)
        self.assertTrue(all(checks.values()))
        self.assertEqual(list(prereg["policies"]), [f"P{index}" for index in range(8)])
        self.assertEqual(prereg["policies"]["P3"]["max_age_bars"], 24)
        self.assertEqual(prereg["policies"]["P4"]["max_age_bars"], 48)
        self.assertFalse(prereg["execution_arms"]["cost_gate"])

    def test_holdout_contract_is_byte_bounded(self):
        prefix = b'{"safe":true}\n'
        suffix = b'{"holdout_price":"MUST_NOT_BE_READ"}\n'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.jsonl"
            path.write_bytes(prefix + suffix)
            loaded = MODULE._read_exact_prefix(
                path, len(prefix), hashlib.sha256(prefix).hexdigest()
            )
        self.assertEqual(loaded, prefix)
        self.assertNotIn(b"holdout_price", loaded)

    def test_semantic_decoder_rejects_holdout_before_market_fields(self):
        row = {
            "schema": "QR_OANDA_HISTORICAL_M5_BA_ROW_V1",
            "instrument": "EUR_USD",
            "granularity": "M5",
            "price_component": "BA",
            "complete": True,
            "time_utc": "2026-05-28T04:05:00.000000Z",
            "volume": 1,
            "bid": {"o": "1", "h": "1", "l": "1", "c": "1"},
            "ask": {"o": "1", "h": "1", "l": "1", "c": "1"},
        }
        with self.assertRaisesRegex(AssertionError, "locked post-tuning"):
            MODULE._parse_bar(
                json.dumps(row).encode("utf-8"),
                "EUR_USD",
                MODULE.parse_time("2026-05-28T04:05:00.000000Z"),
            )


class ChronologyAndGapTests(unittest.TestCase):
    def test_exact_contiguous_rejects_missing_m5(self):
        rows = [bar(0), bar(300), bar(900)]
        self.assertTrue(MODULE._exact_contiguous(rows, 0, 2))
        self.assertFalse(MODULE._exact_contiguous(rows, 0, 3))

    def test_usdjpy_conversion_requires_exact_time_not_prior_close(self):
        start = 1_700_000_000
        bars_by_pair = {
            "EUR_USD": [bar(start), bar(start + 300)],
            "AUD_USD": [bar(start), bar(start + 300)],
            "USD_JPY": [bar(start, 145.0, 0.02)],
        }
        features = {pair: [None] * len(rows) for pair, rows in bars_by_pair.items()}
        market = MODULE.Market(bars_by_pair, features)
        self.assertIsNotNone(market.usd_jpy_rate(start + 300, "c"))
        # A prior close exists, but the exact close/open at T+5 does not.
        self.assertIsNone(market.usd_jpy_rate(start + 600, "c"))
        self.assertIsNone(market.usd_jpy_rate(start + 300, "o"))

    def test_signal_cannot_fill_across_missing_decision_boundary_bar(self):
        start, market = small_market(gapped_eurusd=True)
        decision = start + 300
        source = signal("EUR_USD", "LONG", decision, feature_index=0)
        sim = simulator(market, {decision: [source]})
        sim._record_sources(decision)
        self.assertEqual(sim.source_count, 1)
        # Availability at T+5 is not inspected at decision T.  The pending
        # intent is rejected only when its promised fill timestamp arrives.
        self.assertEqual(sum(map(len, sim.pending_entries.values())), 1)
        sim._process_entries(decision + 300)
        self.assertEqual(sum(map(len, sim.pending_entries.values())), 0)
        self.assertEqual(len(sim.open_positions), 0)
        self.assertEqual(sim.counters["entry_gap_unfilled"], 1)

    def test_entry_and_policy_action_use_strictly_later_open(self):
        start, market = small_market(gapped_eurusd=False)
        decision = start + 300
        source = signal("EUR_USD", "LONG", decision, feature_index=0)
        sim = simulator(market, {decision: [source]})
        sim._record_sources(decision)
        fill = decision + 300
        sim._process_entries(fill)
        self.assertEqual(len(sim.open_positions), 1)
        position = next(iter(sim.open_positions.values()))
        self.assertEqual(position.entry_time, fill)
        self.assertGreater(position.entry_time, position.decision_time)
        sim._request_exit(position, fill, fill + 300, "TEST")
        self.assertEqual(position.scheduled_exit_time, fill + 300)
        with self.assertRaisesRegex(ValueError, "strictly later"):
            sim._request_exit(position, fill, fill, "INVALID")

    def test_open_inventory_gap_is_detected_later_without_backdating(self):
        start, market = small_market(gapped_eurusd=True)
        prereg = MODULE.load_preregistration()
        prereg["splits"]["tuning"] = {
            "from_utc": MODULE.iso_utc(start + 300),
            "to_utc": MODULE.iso_utc(start + 900),
            "use": "fixture",
        }
        sim = MODULE.PolicySimulator(
            "P2", market, {}, calibration(), prereg, "tuning"
        )
        row = MODULE.Position(
            trade_id="gap", policy_id="P2", signal_id="gap",
            pair="EUR_USD", side="LONG", decision_time=start - 600,
            entry_time=start - 300, entry_mid=1.10, entry_observed=1.1001,
            fixed_exit_time=start + 6000, scheduled_exit_time=start + 6000,
            exit_action_time=start - 300, exit_reason="TIME_EXIT_48", tp_pips=4.0,
        )
        sim.open_positions[row.trade_id] = row
        output = sim.run()
        self.assertEqual(output["inventory_samples"][0], 1)
        self.assertEqual(len(output["trades"]), 1)
        trade = output["trades"][0]
        self.assertEqual(trade.exit_reason, "DATA_GAP_TERMINAL_MTM")
        self.assertEqual(trade.price_time, start + 300)
        self.assertEqual(trade.gap_detection_time, start + 600)
        self.assertEqual(trade.exit_time, start + 600)
        self.assertGreater(trade.exit_action_time, trade.price_time)


class EconomicsAndInventoryTests(unittest.TestCase):
    def test_raw_base_adverse_share_lineage_and_cost_order(self):
        start, market = small_market(gapped_eurusd=False)
        sim = simulator(market)
        exit_time = start + 600
        position = MODULE.Position(
            trade_id="t",
            policy_id="P2",
            signal_id="s",
            pair="EUR_USD",
            side="LONG",
            decision_time=start,
            entry_time=start + 300,
            entry_mid=1.1000,
            entry_observed=1.1001,
            fixed_exit_time=exit_time,
            scheduled_exit_time=exit_time,
            exit_action_time=start + 300,
            exit_reason="TIME_EXIT_48",
            tp_pips=4.0,
        )
        sim.open_positions[position.trade_id] = position
        sim._close(position, exit_time, "o")
        trade = sim.trades[0]
        self.assertEqual(trade.signal_id, "s")
        self.assertGreater(trade.raw_pips, trade.base_pips)
        self.assertGreater(trade.base_pips, trade.adverse_pips)
        self.assertAlmostEqual(trade.base_pips - trade.adverse_pips, 1.2, places=9)

    def test_p7_basket_requires_stale_or_trapped_and_nonnegative_usd_group(self):
        start, market = small_market(gapped_eurusd=False)
        when = start + 300
        sim = simulator(market, policy="P7")
        positions = []
        for trade_id, pair, entry_mid, state in (
            ("a", "EUR_USD", 1.0990, "STALE"),
            ("b", "AUD_USD", 0.6590, "HARVEST"),
        ):
            row = MODULE.Position(
                trade_id=trade_id,
                policy_id="P7",
                signal_id=trade_id,
                pair=pair,
                side="LONG",
                decision_time=start - 300,
                entry_time=start,
                entry_mid=entry_mid,
                entry_observed=entry_mid,
                fixed_exit_time=when + 3600,
                scheduled_exit_time=when + 3600,
                exit_action_time=start,
                exit_reason="TIME_EXIT_48",
                tp_pips=4.0,
                last_state=state,
                last_state_time=when,
            )
            positions.append(row)
        groups = sim._basket_unwind(when, positions)
        self.assertEqual(len(groups), 1)
        self.assertEqual({row.trade_id for row in groups[0][2]}, {"a", "b"})
        positions[0].last_state = "HARVEST"
        self.assertEqual(sim._basket_unwind(when, positions), [])

    def test_pair_and_usd_node_caps_are_hard(self):
        start, market = small_market(gapped_eurusd=False)
        sim = simulator(market)
        when = start + 300
        for index in range(4):
            row = MODULE.Position(
                trade_id=f"t{index}", policy_id="P2", signal_id=f"s{index}",
                pair="EUR_USD", side="LONG", decision_time=start - 600,
                entry_time=start - 300, entry_mid=1.10, entry_observed=1.10,
                fixed_exit_time=when + 3000, scheduled_exit_time=when + 3000,
                exit_action_time=start, exit_reason="TIME_EXIT_48", tp_pips=4.0,
            )
            sim.open_positions[row.trade_id] = row
        allowed, reason = sim._cap_allows(signal("EUR_USD", "LONG", when), when)
        self.assertFalse(allowed)
        self.assertEqual(reason, "PAIR_CAP")
        allowed, reason = sim._cap_allows(signal("AUD_USD", "LONG", when), when)
        self.assertFalse(allowed)
        self.assertEqual(reason, "USD_NODE_CAP")

    def test_fill_rechecks_caps_after_inventory_changed_since_decision(self):
        start, market = small_market(gapped_eurusd=False)
        decision = start + 300
        fill_time = decision + 300
        sim = simulator(market)
        for index in range(4):
            row = MODULE.Position(
                trade_id=f"existing{index}", policy_id="P2",
                signal_id=f"existing{index}", pair="EUR_USD", side="LONG",
                decision_time=start - 600, entry_time=start,
                entry_mid=1.10, entry_observed=1.10,
                fixed_exit_time=fill_time + 3000,
                scheduled_exit_time=fill_time + 3000,
                exit_action_time=start, exit_reason="TIME_EXIT_48", tp_pips=4.0,
            )
            sim.open_positions[row.trade_id] = row
        incoming = signal("USD_JPY", "SHORT", decision, feature_index=0)
        sim.pending_entries[fill_time].append(
            MODULE.PendingEntry(incoming, fill_time)
        )
        sim._process_entries(fill_time)
        self.assertEqual(len(sim.open_positions), 4)
        self.assertEqual(sim.counters["fill_cap_skip_usd_node_cap"], 1)

    def test_same_sign_usd_bucket_prevents_later_unwind_net_breach(self):
        start, market = small_market(gapped_eurusd=False)
        when = start + 300
        sim = simulator(market)

        def add(trade_id, pair, side):
            row = MODULE.Position(
                trade_id=trade_id, policy_id="P2", signal_id=trade_id,
                pair=pair, side=side, decision_time=start - 600,
                entry_time=start, entry_mid=market.bar_open(pair, start).mid("o"),
                entry_observed=market.bar_open(pair, start).entry_executable(side),
                fixed_exit_time=when + 3000, scheduled_exit_time=when + 3000,
                exit_action_time=start, exit_reason="TIME_EXIT_48", tp_pips=4.0,
            )
            sim.open_positions[trade_id] = row

        for index in range(4):
            add(f"negative{index}", "EUR_USD", "LONG")
        for index in range(2):
            add(f"positive{index}", "USD_JPY", "LONG")
        # Existing net is -2; another negative lot would leave net -3, but
        # would create a five-lot negative bucket that could later be exposed
        # by closing the two offsetting positive lots.
        allowed, reason = sim._cap_allows(signal("AUD_USD", "LONG", when), when)
        self.assertFalse(allowed)
        self.assertEqual(reason, "USD_NODE_SAME_SIGN_CAP")

    def test_base_is_shared_guard_but_adverse_never_gates(self):
        start, market = small_market(gapped_eurusd=False)
        when = start + 300
        sim = simulator(market)
        row = MODULE.Position(
            trade_id="guarded", policy_id="P2", signal_id="guarded",
            pair="USD_JPY", side="LONG", decision_time=start - 600,
            entry_time=when, entry_mid=145.0, entry_observed=145.01,
            fixed_exit_time=when + 3000, scheduled_exit_time=when + 3000,
            exit_action_time=start, exit_reason="TIME_EXIT_48", tp_pips=4.0,
        )
        sim.open_positions[row.trade_id] = row
        # An adverse-only ruin is evidence, never an execution input.
        sim.realized_jpy["adverse"] = -300_000.0
        self.assertFalse(sim._base_margin_guard(when))
        sim._record_equity(when)
        self.assertIsNone(sim.margin_events["base"])
        self.assertIsNotNone(sim.margin_events["adverse"])
        self.assertEqual(row.scheduled_exit_time, when + 3000)
        # A BASE leverage breach is the single common hard guard.
        sim.realized_jpy["base"] = -195_000.0
        self.assertTrue(sim._base_margin_guard(when))
        self.assertEqual(row.scheduled_exit_time, when + 300)
        self.assertEqual(row.exit_reason, "BASE_MARGIN_HARD_GUARD")

    def test_p3_tp_p5_trapped_and_p6_giveback_schedule_later_open(self):
        start, market = small_market(gapped_eurusd=False)
        when = start + 300
        completed_bar = market.completed("EUR_USD", when)[0]
        aligned = feature(0, when, "LONG")
        market.features_by_pair["EUR_USD"][0] = aligned

        def position(trade_id, entry_mid, tp=1.0):
            return MODULE.Position(
                trade_id=trade_id, policy_id="P3", signal_id=trade_id,
                pair="EUR_USD", side="LONG", decision_time=start - 600,
                entry_time=start - 300, entry_mid=entry_mid,
                entry_observed=entry_mid, fixed_exit_time=when + 6000,
                scheduled_exit_time=when + 6000, exit_action_time=start,
                exit_reason="TIME_EXIT_24", tp_pips=tp,
            )

        p3 = simulator(market, policy="P3")
        pos3 = position("p3", completed_bar.mid("c") - 0.0010)
        p3.open_positions[pos3.trade_id] = pos3
        p3._update_position(pos3, when, completed_bar, aligned)
        p3._dynamic_actions(when, [pos3])
        self.assertEqual(pos3.exit_reason, "RAW_CLOSE_MFE_Q40_TP")
        self.assertEqual(pos3.scheduled_exit_time, when + 300)

        trapped = feature(0, when, "LONG")
        trapped.impulse_side = "SHORT"
        trapped.rail_side = "SHORT"
        trapped.usd_breadth = 1.0  # Opposes long EUR_USD's negative USD node.
        market.features_by_pair["EUR_USD"][0] = trapped
        p5 = simulator(market, policy="P5")
        pos5 = position("p5", completed_bar.mid("c"))
        pos5.policy_id = "P5"
        p5.open_positions[pos5.trade_id] = pos5
        p5._update_position(pos5, when, completed_bar, trapped)
        p5._dynamic_actions(when, [pos5])
        self.assertEqual(pos5.exit_reason, "LGAR_TRAPPED")
        self.assertEqual(pos5.scheduled_exit_time, when + 300)

        market.features_by_pair["EUR_USD"][0] = aligned
        p6 = simulator(market, policy="P6")
        pos6 = position("p6", completed_bar.mid("c") - 0.0004, tp=3.0)
        pos6.policy_id = "P6"
        pos6.tp_reached = True
        pos6.peak_close_mfe_pips = 10.0
        p6.open_positions[pos6.trade_id] = pos6
        p6._update_position(pos6, when, completed_bar, aligned)
        p6._dynamic_actions(when, [pos6])
        self.assertEqual(pos6.exit_reason, "CLOSE_MFE_50PCT_GIVEBACK")
        self.assertEqual(pos6.scheduled_exit_time, when + 300)


class PortfolioMetricTests(unittest.TestCase):
    def test_equity_metrics_use_event_mtm_and_suppress_ruined_month(self):
        jan = MODULE.parse_time("2025-01-31T23:55:00.000000Z")
        feb = MODULE.parse_time("2025-02-01T00:00:00.000000Z")
        metrics = MODULE._equity_metrics(
            [], [(jan, 200_000.0), (feb, 190_000.0), (feb + 300, 210_000.0)],
            "raw", 200_000.0,
        )
        self.assertAlmostEqual(metrics["max_drawdown_fraction"], -0.05)
        self.assertAlmostEqual(metrics["monthly_multiples"]["2025-02"], 1.05)
        ruined = MODULE._equity_metrics(
            [], [(jan, 10_000.0), (feb, -1.0)], "adverse", 200_000.0,
        )
        self.assertTrue(ruined["ruin_observed"])
        self.assertIsNone(ruined["monthly_multiples"]["2025-02"])

    def test_paired_improvement_uses_full_source_portfolio_not_fill_intersection(self):
        prereg = MODULE.load_preregistration()
        start = MODULE.parse_time("2025-01-01T00:00:00.000000Z")
        candidate = {
            "split": "opened_development", "policy_id": "P3",
            "source_signal_count": 100, "source_signal_sha256": "a" * 64,
            "equity_paths": {
                scenario: [(start, 210_000.0), (start + 86400, 220_000.0)]
                for scenario in MODULE.SCENARIOS
            },
            "trades": [], "counters": {},
        }
        baseline = {
            "split": "opened_development", "policy_id": "P2",
            "source_signal_count": 100, "source_signal_sha256": "a" * 64,
            "equity_paths": {
                scenario: [(start, 200_000.0), (start + 86400, 190_000.0)]
                for scenario in MODULE.SCENARIOS
            },
            "trades": [], "counters": {},
        }
        paired = MODULE.paired_improvement(candidate, baseline, prereg)
        self.assertEqual(paired["shared_source_signal_count"], 100)
        self.assertTrue(paired["valid_for_evidence"])
        self.assertIn("no executed-signal intersection", paired["comparison_unit"])
        self.assertEqual(paired["raw"]["terminal_equity_delta_jpy"], 30_000.0)


class ArtifactBoundaryTests(unittest.TestCase):
    def test_replay_has_no_external_runtime_surface(self):
        source = (HERE / "replay_lgar.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertTrue(
            imported.isdisjoint(
                {"http", "urllib", "requests", "socket", "websockets", "subprocess", "keyring"}
            )
        )
        self.assertNotIn("os.environ", source)
        self.assertNotIn("LaunchAgents", source)
        self.assertNotIn("stream-fxtrade", source)

    def test_generated_result_preserves_holdout_and_authority_if_present(self):
        path = HERE / "result.json"
        if not path.exists():
            self.skipTest("result is generated by the deterministic replay")
        result = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(result["holdout"]["price_or_volume_rows_decoded"], 0)
        self.assertEqual(result["holdout"]["labels_computed"], 0)
        self.assertEqual(result["network_attempts"], 0)
        self.assertEqual(result["credential_reads"], 0)
        self.assertEqual(result["external_orders"], 0)


if __name__ == "__main__":
    unittest.main()
