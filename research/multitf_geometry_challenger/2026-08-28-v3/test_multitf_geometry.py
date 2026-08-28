import copy
import importlib.util
import json
import math
import pathlib
import statistics
import unittest


HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "replay_multitf_geometry", HERE / "replay_multitf_geometry.py"
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class MultitfGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prereg = json.loads((HERE / "preregistration.json").read_text())
        cls.prereg_sha = M.sha_file(HERE / "preregistration.json")
        cls.data = M.load_inputs(cls.prereg)
        cls.structures = {
            pair: M.compute_pair_states(pair, rows)
            for pair, rows in cls.data.items()
        }
        cls.states = {
            pair: structure["states"]
            for pair, structure in cls.structures.items()
        }
        M.attach_usd_star(cls.states)
        cls.calibration_ends = {
            pair: int(len(rows) * 0.35) for pair, rows in cls.data.items()
        }
        cls.thresholds = M.derive_thresholds(
            cls.states, cls.calibration_ends, cls.prereg
        )

    def test_contract_inputs_and_v2_are_immutable(self):
        self.assertTrue(all(M.validate_preregistered_contract(self.prereg).values()))
        self.assertEqual(
            M.verify_v2_immutable(self.prereg),
            self.prereg["v2_immutable_hashes"],
        )
        for pair, rows in self.data.items():
            spec = self.prereg["inputs"]["files"][pair]
            self.assertEqual(M.sha_file(pathlib.Path(spec["path"])), spec["sha256"])
            self.assertEqual(len(rows), spec["rows"])
            self.assertTrue(all(row["complete"] for row in rows))

    def test_exact_completed_utc_aggregation(self):
        for pair, structure in self.structures.items():
            rows = self.data[pair]
            for timeframe, minutes in M.PERIOD_MINUTES.items():
                expected_count = minutes // 5
                for bar in structure[timeframe][:100]:
                    self.assertEqual(bar["start"].minute % min(minutes, 60), 0)
                    self.assertEqual(
                        bar["close_time"],
                        bar["start"] + M.dt.timedelta(minutes=minutes),
                    )
                    self.assertEqual(
                        bar["source_end_index"] - bar["source_start_index"] + 1,
                        expected_count,
                    )
                    members = rows[
                        bar["source_start_index"] : bar["source_end_index"] + 1
                    ]
                    self.assertEqual(bar["o"], M.midpoint(members[0], "o"))
                    self.assertEqual(bar["c"], M.midpoint(members[-1], "c"))
                    self.assertEqual(bar["h"], max(M.midpoint(row, "h") for row in members))
                    self.assertEqual(bar["l"], min(M.midpoint(row, "l") for row in members))

    def test_decision_to_fill_is_strictly_later(self):
        pair = "EUR_USD"
        signals = M.make_source_signals(
            pair,
            self.data[pair],
            self.states[pair],
            0,
            1200,
            self.prereg_sha,
        )
        self.assertTrue(signals)
        for signal in signals:
            decision = M.parse_time(signal["decision_time"])
            fill = self.data[pair][signal["fill_index"]]["_time"]
            self.assertGreater(fill, decision)
            self.assertEqual(fill - decision, M.M5_STEP)
            self.assertGreater(signal["fill_index"], signal["decision_index"])

    def test_future_mutation_cannot_change_earlier_states(self):
        pair = "EUR_USD"
        rows = copy.deepcopy(self.data[pair][:1800])
        original = M.compute_pair_states(pair, rows)["states"]
        mutation_time = rows[1500]["_time"]
        for side in ("bid", "ask"):
            for field in ("o", "h", "l", "c"):
                rows[1500][side][field] *= 3.0
        changed = M.compute_pair_states(pair, rows)["states"]
        original_prefix = [
            M.state_feature_hash(state)
            for state in original
            if state["decision_time_dt"] <= mutation_time
        ]
        changed_prefix = [
            M.state_feature_hash(state)
            for state in changed
            if state["decision_time_dt"] <= mutation_time
        ]
        self.assertEqual(original_prefix, changed_prefix)

    def test_spread_does_not_change_prediction_features(self):
        pair = "EUR_USD"
        rows = copy.deepcopy(self.data[pair][:1800])
        widened = copy.deepcopy(rows)
        for row in widened:
            for field in ("o", "h", "l", "c"):
                mid = M.midpoint(row, field)
                row["bid"][field] = mid - 0.01
                row["ask"][field] = mid + 0.01
        first = M.compute_pair_states(pair, rows)["states"]
        second = M.compute_pair_states(pair, widened)["states"]
        self.assertEqual(
            [M.state_feature_hash(state) for state in first],
            [M.state_feature_hash(state) for state in second],
        )

    def test_graph_requires_same_timestamp_all_three_pairs(self):
        copied = {pair: [copy.deepcopy(state) for state in states[:20]] for pair, states in self.states.items()}
        # Different starting clocks mean no forward fill is permitted.
        M.attach_usd_star(copied)
        for pair, states in copied.items():
            for state in states:
                if state["usd_star"] is not None:
                    members = [
                        candidate
                        for other in copied.values()
                        for candidate in other
                        if candidate["decision_time"] == state["decision_time"]
                    ]
                    self.assertEqual({item["pair"] for item in members}, {"EUR_USD", "AUD_USD", "USD_JPY"})

    def test_calibration_thresholds_are_future_independent(self):
        pair = "EUR_USD"
        changed = {name: [copy.deepcopy(state) for state in states] for name, states in self.states.items()}
        for state in changed[pair]:
            if state["decision_source_end_index"] >= self.calibration_ends[pair]:
                state["h1_range_ratio"] *= 1000.0
                state["h4_extension"] *= 1000.0
                if state["usd_residual"] is not None:
                    state["usd_residual"] *= 1000.0
        thresholds = M.derive_thresholds(changed, self.calibration_ends, self.prereg)
        self.assertEqual(thresholds[pair], self.thresholds[pair])

    def test_exact_eight_config_gates(self):
        self.assertEqual(sorted(self.prereg["configs"]), [f"C{i}" for i in range(8)])
        threshold = {
            "h1_compression_q35": 0.01,
            "h4_extension_q65": 1.0,
            "usd_residual_abs_q65": 0.001,
        }
        base = {
            "h4_side": "LONG",
            "h1_side": "LONG",
            "h1_range_ratio": 0.005,
            "h1_deceleration": True,
            "h4_extension": 2.0,
            "pullback_side": "LONG",
            "acceptance_side": "LONG",
            "sweep_side": "LONG",
            "usd_residual": 0.002,
            "graph_side": "LONG",
        }
        self.assertEqual(M.gate_signal("C0", base, threshold), "LONG")
        self.assertEqual(M.gate_signal("C1", base, threshold), "LONG")
        self.assertEqual(M.gate_signal("C2", base, threshold), "LONG")
        self.assertEqual(M.gate_signal("C3", base, threshold), "LONG")
        self.assertEqual(M.gate_signal("C4", base, threshold), "LONG")
        self.assertEqual(M.gate_signal("C5", base, threshold), "LONG")
        fade = dict(base, sweep_side="SHORT")
        self.assertEqual(M.gate_signal("C6", fade, threshold), "SHORT")
        self.assertEqual(M.gate_signal("C7", base, threshold), "LONG")

    def test_family_corrected_currency_time_cluster_math(self):
        trades = []
        for day, values in (("01", [1.0, 3.0]), ("02", [2.0, 6.0]), ("03", [4.0, 8.0])):
            for index, value in enumerate(values):
                trades.append(
                    {
                        "decision_time": f"2026-01-{day}T0{index}:00:00Z",
                        "raw_pips": value,
                    }
                )
        summary = M.cluster_summary(trades, "raw", self.prereg["selection"])
        means = [2.0, 4.0, 6.0]
        z_value = statistics.NormalDist().inv_cdf(1.0 - 0.05 / 8.0)
        standard_error = statistics.stdev(means) / math.sqrt(3)
        self.assertEqual(summary["n_eff_currency_time_clusters"], 3)
        self.assertAlmostEqual(summary["cluster_mean_pips"], 4.0)
        self.assertAlmostEqual(summary["family_adjusted_lcb_pips"], 4.0 - z_value * standard_error)

    def test_split_boundary_and_arm_lineage(self):
        pair = "EUR_USD"
        end = 2400
        signals = M.make_source_signals(
            pair,
            self.data[pair],
            self.states[pair],
            1200,
            end,
            self.prereg_sha,
        )
        replay = M.replay_config(
            pair,
            self.data[pair],
            signals,
            self.thresholds[pair],
            "C4",
            end,
            self.prereg,
        )
        self.assertTrue(replay["trades"])
        for trade in replay["trades"]:
            self.assertLess(trade["exit_index"], end)
            self.assertLessEqual(trade["age_bars"], 48)
            self.assertAlmostEqual(
                trade["base_pips"] - trade["adverse_pips"], 1.2
            )
            self.assertEqual(
                trade["trade_id"],
                M.sha_bytes(
                    f"{trade['signal_id']}|C4|{trade['side']}".encode("utf-8")
                ),
            )

    def test_raw_selection_cannot_be_changed_by_cost_arms(self):
        configs = {}
        for index in range(8):
            raw = {
                "family_adjusted_lcb_pips": float(index),
                "expectancy_pips": float(index),
                "pair_min_expectancy_pips": float(index),
                "session_min_expectancy_pips": float(index),
                "n_eff_currency_time_clusters": 30,
                "trades": 100,
                "pairs_meeting_trade_floor": 3,
                "pairs_with_positive_expectancy": 3,
                "sessions_with_positive_expectancy": 3,
            }
            configs[f"C{index}"] = {
                "tuning": {
                    "scenario_metrics": {
                        "raw": raw,
                        "base": {"expectancy_pips": 1000.0 - index},
                        "adverse": {"expectancy_pips": -1000.0 + index},
                    }
                }
            }
        selected = M.select_config(configs, [f"C{i}" for i in range(8)], self.prereg["selection"])[0]
        self.assertEqual(selected, "C7")
        for index in range(8):
            configs[f"C{index}"]["tuning"]["scenario_metrics"]["base"]["expectancy_pips"] = index * 1e9
        self.assertEqual(
            M.select_config(configs, [f"C{i}" for i in range(8)], self.prereg["selection"])[0],
            selected,
        )

    def test_full_result_is_development_only_and_zero_authority(self):
        result = M.main(write=False)
        self.assertFalse(result["admission"])
        self.assertFalse(result["holdout"])
        self.assertFalse(result["shadow_challenger_eligible"])
        self.assertTrue(result["profit_unproven"])
        self.assertEqual(result["network_attempts"], 0)
        self.assertEqual(result["credential_reads"], 0)
        self.assertEqual(result["external_orders"], 0)
        self.assertEqual(sorted(result["configs"]), [f"C{i}" for i in range(8)])
        selected = result["selected_config"]
        for period in ("tuning", "opened_development"):
            metrics = result["configs"][selected][period]["scenario_metrics"]
            self.assertEqual(
                metrics["raw"]["lineage_sha256"],
                metrics["base"]["lineage_sha256"],
            )
            self.assertEqual(
                metrics["raw"]["lineage_sha256"],
                metrics["adverse"]["lineage_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
